import asyncio
import logging
from time import sleep

from moviepy import VideoFileClip
from telegram.config import WHISPER_CLI,  WHISPER_MODEL
from pathlib import Path
import subprocess
from langfuse.decorators import observe
import tempfile
import shutil

@observe()
async def convert_voice(input_path: Path) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = Path(tmp.name)
    command = [
    "ffmpeg",
    "-y", # перезаписывать если такой файл существует
    "-i", str(input_path),
    "-ar", "16000",
    "-ac", "1", # установить 1 канал (моно).
    "-af", "highpass=f=200, lowpass=f=3000, dynaudnorm, afftdn",
#  highpass=f=200 — отрезаем шумы ниже 200 Hz.
# lowpass=f=3000 — отрезаем мусор выше 3000 Hz.
# dynaudnorm — нормализуем громкость (голос будет ровный по громкости).
# afftdn — адаптивное подавление шума (очень круто для чистоты речи).
    "-sample_fmt", "s16",
    # 16-битное аудио (PCM формат, для Whisper идеально).
    str(output_path)
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {stderr.decode()}")
    new_path = input_path.with_suffix('.wav')
    shutil.move(str(output_path), str(new_path))
    return new_path

@observe()
async def extract_audio(input_file: Path) -> str:
    video = VideoFileClip(input_file)
    output_file = input_file.with_suffix('.wav')
    video.audio.write_audiofile(output_file, codec='pcm_s16le')
    return output_file




