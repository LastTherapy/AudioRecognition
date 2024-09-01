import logging
from aiogram import Dispatcher
from aiogram.types import Message
from aiogram import F
from utils import download_file, perform_voice_recognition

def setup_handlers(dp: Dispatcher):
    @dp.message(F.content_type.in_({'voice'}))
    async def auto_voice_recognition(message: Message):
        logging.info(f'Voice received from {message.from_user.full_name} with id {message.from_user.id}')
        await perform_voice_recognition(message, model='large')
