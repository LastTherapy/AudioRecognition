import logging
from aiogram import Dispatcher
from aiogram.types import Message
from aiogram import F
from aiogram.filters.command import Command
import gc
from utils import  perform_voice_recognition, extract_audio

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
        audio_path: str = await extract_audio(message)
        await perform_voice_recognition(message, model='large', audio_path=audio_path)

    @dp.message(F.content_type.in_({'voice'}))
    async def auto_voice_recognition(message: Message):
        logging.info(f'Voice received from {message.from_user.full_name} with id {message.from_user.id}')
        await perform_voice_recognition(message, model='large')
        gc.collect()


    @dp.message()
    async def log_text_messages(message: Message):
        print(f"{message.chat.full_name}: {message.text}")
