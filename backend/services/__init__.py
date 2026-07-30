"""Service layer: one module per pipeline concern.

audio          ffmpeg conversion to 16 kHz mono PCM
transcription  Whisper large-v3 + silero VAD (the logic from the original app.py)
translation    Arabic -> English through the LLM
classification theme / issue / reason through the LLM
excel          report writer
pipeline       stitches the four stages together for a single call
job_manager    queue + worker pool that runs many calls concurrently
job_store      persistence for job and call state
"""
