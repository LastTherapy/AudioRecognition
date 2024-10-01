import logging
from aiogram import Dispatcher
from aiogram.types import Message
from aiogram import F
from aiogram.filters.command import Command

from utils import  perform_voice_recognition

def setup_handlers(dp: Dispatcher):
    @dp.message(Command("start"))
    async def start_message(message: Message):
        await message.answer("Просто отправьте голосовое сообщение для распознавания и подождите результата.")

    @dp.message(Command("help"))
    async def help_message(message: Message):
        await message.answer("Бот распознает голосовые сообощения. Отправьте ему их лично, либо просто добавьте бота в группу.  Для распознавания используется open-source модель whisper.")

    @dp.message(F.content_type.in_({'voice'}))
    async def auto_voice_recognition(message: Message):
        logging.info(f'Voice received from {message.from_user.full_name} with id {message.from_user.id}')
        await perform_voice_recognition(message, model='large')


    @dp.message()
    async def log_text_messages(message: Message):
        print(f"{message.chat.full_name}: {message.text}")
