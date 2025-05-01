import asyncio
import logging
from time import sleep
from aiogram import Bot
from aiogram.types import Message
from typing import List
from moviepy import VideoFileClip
from telegram.config import WHISPER_CLI,  WHISPER_MODEL
from pathlib import Path
import subprocess
import re



# # Загрузим необходимые ресурсы (раз в проекте):
# nltk.download('punkt')          # токенизатор предложений/слов
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# model = SentenceTransformer('all-MiniLM-L6-v2')  # лёгкая и быстрая модель


# async def convert_voice(input_path: Path) -> Path:
#     output_path: Path = input_path.with_suffix('.wav')
#     command = [
#     "ffmpeg",
#     "-i", str(input_path),
#     "-ar", "16000",
#     "-ac", "1", # установить 1 канал (моно).
#     "-af", "highpass=f=200, lowpass=f=3000, dynaudnorm, afftdn",
# #  highpass=f=200 — отрезаем шумы ниже 200 Hz.
# # lowpass=f=3000 — отрезаем мусор выше 3000 Hz.
# # dynaudnorm — нормализуем громкость (голос будет ровный по громкости).
# # afftdn — адаптивное подавление шума (очень круто для чистоты речи).
#     "-sample_fmt", "s16",
#     # 16-битное аудио (PCM формат, для Whisper идеально).
#     str(output_path)
#     ]
#     process = await asyncio.create_subprocess_exec(
#         *command,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE
#     )
#     stdout, stderr = await process.communicate()
#     if process.returncode != 0:
#         raise RuntimeError(f"FFmpeg conversion failed: {stderr.decode()}")
    
#     return output_path

# async def extract_audio(input_file: Path) -> str:
#     video = VideoFileClip(input_file)
#     output_file = input_file.with_suffix('.wav')
#     video.audio.write_audiofile(output_file, codec='pcm_s16le')
#     return output_file


# async def whisper_cli_recognition(input_path: Path) -> Path:
#     output_path = input_path.with_suffix('')

#     command = [
#         WHISPER_CLI,
#         '-m', WHISPER_MODEL,
#         '-f', str(input_path),
#         '-l', 'ru',
#         '-of', str(output_path),
#         '-otxt'
#     ]

#     print(f"Запускаю команду: {' '.join(command)}")

#     process = await asyncio.create_subprocess_exec(
#         *command,
#         stdout=asyncio.subprocess.PIPE,
#         stderr=asyncio.subprocess.PIPE,
#     )

#     stdout, stderr = await process.communicate()

#     print(f"[whisper-cli stdout]:\n{stdout.decode()}")
#     print(f"[whisper-cli stderr]:\n{stderr.decode()}")

#     if process.returncode != 0:
#         raise RuntimeError(f"Распознавание провалилось с ко дом {process.returncode}")

#     return output_path.with_suffix('.txt')

# async def ml_split_text(raw_text: str) -> str:
#     # 1) Разбиваем на предложения
#     sentences = nltk.tokenize.sent_tokenize(raw_text)
#     if len(sentences) < 2:
#         return raw_text  # слишком короткий текст

#     # 2) Получаем эмбеддинги
#     emb = model.encode(sentences, convert_to_numpy=True)
#     # 3) Считаем «скачки» смысловой дистанции
#     dists = [cosine(emb[i], emb[i+1]) for i in range(len(emb)-1)]
#     # 4) Авто-порог: среднее + 1σ
#     mu, sigma = np.mean(dists), np.std(dists)
#     threshold = mu + sigma
#     break_idxs = [i for i, d in enumerate(dists) if d > threshold]
#     # 5) Формируем параграфы по найденным границам
#     paras, start = [], 0
#     for idx in break_idxs:
#         paras.append(" ".join(sentences[start:idx+1]))
#         start = idx+1
#     paras.append(" ".join(sentences[start:]))
#     # 6) Гарантируем минимум 2 параграфа
#     if len(paras) == 1:
#         mid = len(sentences) // 2 or 1
#         paras = [
#             " ".join(sentences[:mid]),
#             " ".join(sentences[mid:])
#         ]
#     # 7) Возвращаем единую строку с двойными переводами строк
#     return "\n\n".join(p.strip() for p in paras)


# async def clean_whisper_text_basic(raw_text: str) -> str:
#     # Убираем переносы строк, которые НЕ после точки, вопроса или восклицания
#     text = re.sub(r'(?<![\.\?\!])\n', ' ', raw_text)
#     # Заменяем множественные пробелы на один
#     text = re.sub(r'\s+', ' ', text)
#     # Убираем пробелы перед пунктуацией
#     text = re.sub(r' \.', '.', text)
#     text = re.sub(r' \?', '?', text)
#     text = re.sub(r' \!', '!', text)
#     return text.strip()


def split_text_for_telegram(text: str, max_length: int = 4096) -> List[str]:
    """
    Разбивает текст на части, каждая из которых не превышает max_length символов.
    Сначала пытается разделить по абзацам, затем по предложениям, и, при необходимости, по символам.
    """
    chunks = []
    paragraphs = text.split('\n\n')  # Разделение по абзацам

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) <= max_length:
            chunks.append(paragraph)
        else:
            # Разделение абзаца на предложения
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            current_chunk = ''
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= max_length:
                    current_chunk += (' ' if current_chunk else '') + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    if len(sentence) <= max_length:
                        current_chunk = sentence
                    else:
                        # Разделение длинного предложения на части
                        for i in range(0, len(sentence), max_length):
                            chunks.append(sentence[i:i+max_length])
                        current_chunk = ''
            if current_chunk:
                chunks.append(current_chunk)

    return chunks