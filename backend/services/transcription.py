"""Speech to text.

The Whisper + silero-VAD logic is carried over from the original app.py
essentially unchanged - chunking on VAD boundaries, per-segment quality gates,
silence flags - because that part was already doing the right thing. What is
new is the wrapping:

  * heavy imports (torch/whisper/librosa/silero) happen inside the backend
    class, so STT_BACKEND=mock runs on a laptop with none of them installed
  * the model is loaded exactly once, behind a lock
  * transcription runs in a bounded ThreadPoolExecutor, so the GPU sees at
    most STT_CONCURRENCY files at a time while the event loop stays free
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from backend.core.config import Settings
from backend.core.errors import StageError
from backend.core.logging_config import get_logger
from backend.schemas.theme import TranscriptionResult

logger = get_logger(__name__)

NO_SPEECH_MARKER = "[No speech detected]"


class TranscriberBackend(Protocol):
    def load(self) -> None: ...
    def is_loaded(self) -> bool: ...
    def transcribe_file(self, wav_path: Path) -> dict[str, Any]: ...


# ==========================================================================
# Whisper large-v3
# ==========================================================================
class WhisperTranscriber:
    """Ported from app.py. Runs on the STT thread pool - never on the loop."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self.chunk_duration = settings.CHUNK_DURATION
        self.silence_pad = settings.SILENCE_PAD
        self.target_sample_rate = settings.TARGET_SAMPLE_RATE
        self.initial_prompt = settings.WHISPER_INITIAL_PROMPT or ""

        self._model: Any = None
        self._vad_model: Any = None
        self._load_lock = threading.Lock()
        self._np: Any = None  # numpy, imported on load

    # ---- lifecycle ---------------------------------------------------------
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return

            import numpy as np
            import torch
            import whisper
            from silero_vad import load_silero_vad

            self._np = np
            device = torch.device(self._settings.WHISPER_DEVICE)
            model_ref = self._settings.WHISPER_MODEL_PATH

            logger.info("Loading Whisper '%s' on %s ...", model_ref, device)
            started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            self._model = whisper.load_model(model_ref, device=device)
            self._vad_model = load_silero_vad()
            logger.info(
                "Whisper + silero VAD loaded in %.1fs", time.perf_counter() - started
            )

    # ---- utils -------------------------------------------------------------
    @staticmethod
    def format_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.2f}"

    # ---- audio -------------------------------------------------------------
    def load_audio(self, wav_path: Path):
        import wave

        import librosa

        np = self._np
        with wave.open(str(wav_path), "rb") as wav:
            sr = wav.getframerate()
            n_channels = wav.getnchannels()
            frames = wav.readframes(wav.getnframes())

        audio_np = np.frombuffer(frames, dtype=np.int16)
        if audio_np.size == 0:
            return audio_np, 0.0

        if n_channels > 1:
            audio_np = audio_np.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

        audio_np = audio_np.astype(np.float32) / 32768.0
        if sr != self.target_sample_rate:
            audio_np = librosa.resample(
                audio_np, orig_sr=sr, target_sr=self.target_sample_rate
            )

        audio_np = (audio_np * 32768.0).astype(np.int16)
        return audio_np, len(audio_np) / self.target_sample_rate

    # ---- VAD ---------------------------------------------------------------
    def get_full_speech_timestamps(self, audio_np) -> list[dict[str, int]]:
        from silero_vad import get_speech_timestamps

        audio_float = audio_np.astype(self._np.float32) / 32768.0
        return get_speech_timestamps(
            audio_float, self._vad_model, sampling_rate=self.target_sample_rate
        )

    def get_silence_segments_full_audio(self, audio_np, speech_timestamps):
        silence_segments = []
        prev_end = 0
        for seg in speech_timestamps:
            if seg["start"] > prev_end:
                silence_segments.append((prev_end, seg["start"]))
            prev_end = seg["end"]
        if prev_end < len(audio_np):
            silence_segments.append((prev_end, len(audio_np)))
        return silence_segments

    def merge_silences(self, silence_segments, max_gap_sec: float = 1.0):
        if not silence_segments:
            return []
        merged = []
        cur_start, cur_end = silence_segments[0]
        for start, end in silence_segments[1:]:
            gap = (start - cur_end) / self.target_sample_rate
            if gap <= max_gap_sec:
                cur_end = end
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        merged.append((cur_start, cur_end))
        return merged

    def compute_silence_flags(self, audio_np, speech_timestamps) -> dict[str, bool]:
        silence_segments = self.merge_silences(
            self.get_silence_segments_full_audio(audio_np, speech_timestamps)
        )

        five_sec_flag = True
        thirty_sec_flag = False
        two_min_flag = False

        for start, end in silence_segments:
            duration = (end - start) / self.target_sample_rate
            start_sec = start / self.target_sample_rate
            if start_sec <= 0 and duration < 5:
                five_sec_flag = False
            if duration >= 30:
                thirty_sec_flag = True
            if duration >= 120:
                two_min_flag = True

        return {
            "5_sec_flag": five_sec_flag,
            "30_sec_flag": thirty_sec_flag,
            "2_min_flag": two_min_flag,
        }

    # ---- chunking ----------------------------------------------------------
    def create_chunks_from_speech(self, audio_np, speech_timestamps):
        """Group VAD segments into ~chunk_duration blocks, splitting only at a
        silence gap so no word is ever cut in half."""
        if not speech_timestamps:
            return [], []

        pad = int(self.silence_pad * self.target_sample_rate)
        chunk_size_samples = int(self.chunk_duration * self.target_sample_rate)
        audio_len = len(audio_np)

        chunks: list[Any] = []
        time_ranges: list[tuple[float, float]] = []

        def flush(g_start: int, g_end: int) -> None:
            s = max(0, g_start - pad)
            e = min(audio_len, g_end + pad)
            chunks.append(audio_np[s:e])
            time_ranges.append(
                (s / self.target_sample_rate, e / self.target_sample_rate)
            )

        group_start = speech_timestamps[0]["start"]
        group_end = speech_timestamps[0]["end"]

        for seg in speech_timestamps[1:]:
            if seg["end"] - group_start > chunk_size_samples:
                flush(group_start, group_end)
                group_start, group_end = seg["start"], seg["end"]
            else:
                group_end = seg["end"]

        flush(group_start, group_end)
        return chunks, time_ranges

    # ---- transcription -----------------------------------------------------
    def transcribe_chunk(self, chunk, chunk_start_time: float) -> dict[str, Any]:
        chunk_float = chunk.astype(self._np.float32) / 32768.0
        language = self._settings.WHISPER_LANGUAGE.strip().lower()

        result = self._model.transcribe(
            chunk_float,
            verbose=False,
            language=None if language in ("", "auto") else language,
            beam_size=5,
            best_of=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

        segments = []
        for seg in result.get("segments", []):
            # Whisper hallucinates on silence; these three gates drop most of it.
            if seg.get("no_speech_prob", 0) > 0.6:
                continue
            if seg.get("avg_logprob", 0) < -1.0:
                continue
            if seg.get("compression_ratio", 0) > 2.4:
                continue
            segments.append(
                {
                    "start": self.format_time(chunk_start_time + seg["start"]),
                    "end": self.format_time(chunk_start_time + seg["end"]),
                    "text": seg["text"].strip(),
                }
            )

        return {
            "main_transcription": (
                " ".join(s["text"] for s in segments) if segments else NO_SPEECH_MARKER
            ),
            "segments": segments,
            "language": result.get("language", language or "unknown"),
        }

    def transcribe_file(self, wav_path: Path) -> dict[str, Any]:
        self.load()
        audio_np, total_duration_seconds = self.load_audio(wav_path)

        if len(audio_np) == 0:
            return _empty_file_data(str(wav_path), 0.0, self.format_time)

        speech_timestamps = self.get_full_speech_timestamps(audio_np)
        silence_flags = self.compute_silence_flags(audio_np, speech_timestamps)
        chunks_data, chunk_time_ranges = self.create_chunks_from_speech(
            audio_np, speech_timestamps
        )

        file_data: dict[str, Any] = {
            "file": str(wav_path),
            "total_duration": self.format_time(total_duration_seconds),
            "total_duration_seconds": total_duration_seconds,
            "silence_flags": silence_flags,
            "language": "unknown",
            "chunks": [],
        }

        if not chunks_data:
            file_data["chunks"].append(
                {
                    "start": self.format_time(0),
                    "end": self.format_time(total_duration_seconds),
                    "main_transcription": NO_SPEECH_MARKER,
                    "segments": [],
                }
            )
            return file_data

        languages: list[str] = []
        for index, chunk in enumerate(chunks_data):
            start_sec, end_sec = chunk_time_ranges[index]
            logger.info("Transcribing chunk %d/%d", index + 1, len(chunks_data))
            chunk_result = self.transcribe_chunk(chunk, start_sec)
            if chunk_result["language"]:
                languages.append(chunk_result["language"])
            file_data["chunks"].append(
                {
                    "start": self.format_time(start_sec),
                    "end": self.format_time(end_sec),
                    "main_transcription": chunk_result["main_transcription"],
                    "segments": chunk_result["segments"],
                }
            )

        if languages:
            file_data["language"] = Counter(languages).most_common(1)[0][0]
        return file_data


# ==========================================================================
# Mock backend - lets the API, queue, report and UI be exercised with no GPU
# ==========================================================================
class MockTranscriber:
    _SAMPLE = (
        "Agent: Good morning, thank you for calling. How may I help you? "
        "Customer: My credit card was declined at a hotel abroad yesterday and "
        "I could not pay. Agent: I can see the risk system blocked the card "
        "after an international attempt. I will remove the block now. "
        "Customer: This is the second time I am calling about this. "
        "Agent: I apologise, the block is now removed and travel notification "
        "is active for thirty days."
    )

    def __init__(self, settings: Settings):
        self._settings = settings

    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def transcribe_file(self, wav_path: Path) -> dict[str, Any]:
        time.sleep(0.4)  # pretend to do work so progress is visible in the UI
        return {
            "file": str(wav_path),
            "total_duration": "00:01:30.00",
            "total_duration_seconds": 90.0,
            "silence_flags": {"5_sec_flag": False, "30_sec_flag": False, "2_min_flag": False},
            "language": "en",
            "chunks": [
                {
                    "start": "00:00:00.00",
                    "end": "00:01:30.00",
                    "main_transcription": self._SAMPLE,
                    "segments": [
                        {"start": "00:00:00.00", "end": "00:01:30.00", "text": self._SAMPLE}
                    ],
                }
            ],
        }


def _empty_file_data(path: str, duration: float, fmt) -> dict[str, Any]:
    return {
        "file": path,
        "total_duration": fmt(duration),
        "total_duration_seconds": duration,
        "silence_flags": {"5_sec_flag": True, "30_sec_flag": False, "2_min_flag": False},
        "language": "unknown",
        "chunks": [
            {
                "start": fmt(0),
                "end": fmt(duration),
                "main_transcription": NO_SPEECH_MARKER,
                "segments": [],
            }
        ],
    }


# ==========================================================================
# Async service wrapper
# ==========================================================================
class TranscriptionService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._backend: TranscriberBackend = (
            MockTranscriber(settings)
            if settings.STT_BACKEND.lower() == "mock"
            else WhisperTranscriber(settings)
        )
        # max_workers == STT_CONCURRENCY is what actually bounds GPU usage;
        # everything else queues here instead of fighting for VRAM.
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, settings.STT_CONCURRENCY),
            thread_name_prefix="stt",
        )

    @property
    def backend_name(self) -> str:
        return self._settings.STT_BACKEND

    @property
    def model_loaded(self) -> bool:
        return self._backend.is_loaded()

    async def warmup(self) -> None:
        """Pay the model-load cost at startup instead of on the first call."""
        if not self._settings.STT_WARMUP:
            return
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._backend.load)
        except Exception:
            # A missing GPU should degrade /health/ready, not stop the API.
            logger.exception("STT warmup failed - transcription will error per call")

    async def transcribe(self, wav_path: Path, json_out: Path | None = None) -> TranscriptionResult:
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(
                self._executor, self._backend.transcribe_file, wav_path
            )
        except StageError:
            raise
        except Exception as exc:
            raise StageError(
                "transcribing", f"Whisper failed on {wav_path.name}: {exc}"
            ) from exc

        if json_out is not None:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        text = build_transcript_paragraph(data)
        return TranscriptionResult(
            text=text,
            srt=build_srt(data),
            language=data.get("language", "unknown") or "unknown",
            duration_seconds=float(data.get("total_duration_seconds", 0.0) or 0.0),
            segment_count=sum(len(c.get("segments", [])) for c in data.get("chunks", [])),
            silence_flags=data.get("silence_flags", {}),
            json_path=str(json_out) if json_out else None,
            no_speech=not text.strip(),
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


# ---- transcript formatting -------------------------------------------------
def build_transcript_paragraph(data: dict[str, Any]) -> str:
    """Every chunk's text as one flowing paragraph, minus the no-speech marker."""
    parts = [
        chunk.get("main_transcription", "")
        for chunk in data.get("chunks", [])
        if chunk.get("main_transcription")
        and chunk["main_transcription"] != NO_SPEECH_MARKER
    ]
    return " ".join(parts).strip()


def build_srt(data: dict[str, Any]) -> str:
    """Timestamped lines - this is what the translation prompt consumes."""
    lines: list[str] = []
    for chunk in data.get("chunks", []):
        for segment in chunk.get("segments", []):
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"{segment['start']} -- {segment['end']} : {text}")
    return "\n".join(lines)
