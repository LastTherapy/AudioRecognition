import logging
import re
from aiogram import Dispatcher
from aiogram.types import Message
from aiogram import F
from aiogram.filters.command import Command
from typing import List
from telegram.utils import convert_voice, whisper_cli_recognition, clean_whisper_text_basic, extract_audio, ml_split_text
from pathlib import Path
from telegram.config import VOICE_STORAGE, VIDEO_STORAGE

def setup_handlers(dp: Dispatcher):
    @dp.message(Command("start"))
    async def start_message(message: Message):
        await message.answer("Просто отправьте голосовое сообщение для распознавания и подождите результата.")

    @dp.message(Command("help"))
    async def help_message(message: Message):
        await message.answer("Бот распознает голосовые сообощения. Отправьте ему их лично, либо просто добавьте бота в группу.  Для распознавания используется open-source модель whisper.")

    @dp.message(F.video_note)
    async def video_note_handler(message: Message):
        logging.info(f"Video note received from {message.from_user.full_name} with id {message.from_user.id}")
        try:
            file = await message.bot.get_file(message.video_note.file_id, request_timeout=30)
            destination = (Path(VIDEO_STORAGE) / str(message.from_user.id) / str(message.message_id)).with_suffix('.mp4')
            destination.parent.mkdir(parents=True, exist_ok=True)
            await message.bot.download_file(file.file_path, destination, timeout=45)
        except TimeoutError:
            await message.answer("Не получилось скачать файл")
            
        
        audio_path: Path = await extract_audio(destination)
        wav_path = await convert_voice(audio_path)
        txt_path = await whisper_cli_recognition(wav_path)
        try:
            text = txt_path.read_text(encoding='utf-8')
        except Exception as e:
            logging.error(f"Ошибка чтения файла распознавания: {e}")
            await message.answer("Ошибка при чтении результата распознавания")
            return
        
        if text.strip():
            text = await clean_whisper_text_basic(text)
            text = await ml_split_text(text)
            if len(text) < 4096:
                await message.answer(text)
            else:
                chunks = split_text_for_telegram(text)
                for chunk in chunks:
                    message.answer(chunk)            
        else:
            pass 
        

    @dp.message(F.voice)
    async def auto_voice_recognition(message: Message):
        logging.info(f'Voice received from {message.from_user.full_name} with id {message.from_user.id}')
        asnwer_message: Message = await message.bot.send_message(message.chat.id, "Скачиваю файл")
        try:
            file = await message.bot.get_file(message.voice.file_id, request_timeout=30)
            destination = (Path(VOICE_STORAGE) / str(message.from_user.id) / str(message.message_id)).with_suffix('.ogg')
            destination.parent.mkdir(parents=True, exist_ok=True)
            await message.bot.download_file(file.file_path, timeout=45, destination=destination)
            await asnwer_message.edit_text('Успех. Удаляю шумы.')

        except TimeoutError:
            await asnwer_message.edit_text("Не получилось скачать файл")
            return
        
         # Конвертация ogg → wav
        wav_path = await convert_voice(destination)
        await asnwer_message.edit_text('Успех. Слушаю аудио.')
        # Распознавание через Whisper
        txt_path = await whisper_cli_recognition(wav_path)
        await asnwer_message.edit_text('Расставляю абзацы')
        try:
            text = txt_path.read_text(encoding='utf-8')
        except Exception as e:
            logging.error(f"Ошибка чтения файла распознавания: {e}")
            await message.answer("Ошибка при чтении результата распознавания")
            return

        # Отправляем текст пользователю
        if text.strip():
            text = await clean_whisper_text_basic(text)
            text = await ml_split_text(text)
            if len(text) < 4096:
                await asnwer_message.edit_text(text)
            else:
                await asnwer_message.delete()
                chunks = split_text_for_telegram(text)
                for chunk in chunks:
                    message.answer(chunk)
                    
        else:
            await asnwer_message.edit_text("Аудио без слов")


    @dp.message()
    async def log_text_messages(message: Message):
        logging.info(f"{message.chat.full_name}: {message.text}")



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