from pathlib import Path
import asyncio
import logging
from audio_recognition.config import WHISPER_CLI, WHISPER_MODEL
from langfuse.decorators import observe


@observe()
async def whisper_cli_recognition(input_path: Path) -> Path:
    output_path = input_path.with_suffix('')

    command = [
        WHISPER_CLI,
        '-m', WHISPER_MODEL,
        '-f', str(input_path),
        '-l', 'ru',
        '-of', str(output_path),
        '-otxt'
    ]

    print(f"Запускаю команду: {' '.join(command)}")

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    print(f"[whisper-cli stdout]:\n{stdout.decode()}")
    print(f"[whisper-cli stderr]:\n{stderr.decode()}")
    
    try:
        text = output_path.with_suffix('.txt').read_text(encoding='utf-8')
    except Exception as e:
        logging.error(f"Ошибка чтения файла распознавания: {e}")
        return

    return text
