import os
import logging
from aiogram import Bot
from aiogram.types import Message
from config import VOICE_STORAGE
import WhisperRecognition


async def download_file(bot: Bot, message: Message) -> str:
    file = await bot.get_file(message.voice.file_id)
    destination_file = os.path.join(VOICE_STORAGE, f'{message.message_id}.oga')
    await bot.download_file(file.file_path, destination_file)
    return destination_file


async def perform_voice_recognition(message: Message, model: str = 'small'):
    bot = message.bot
    destination_file = await download_file(bot, message)

    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    recognized: Message = await message.reply(f"Начинаю распознавание... Использую {model} модель...")

    try:
        result = WhisperRecognition.recognition(destination_file, model)
    except Exception as e:
        logging.exception("Error in voice recognition")
        result = "Sorry, no more GPU memory available just now. Try again later."

    if not result:
        result = "Sorry, no text in voice recognition."

    if len(result) < 4096:
        await bot.edit_message_text(result, chat_id=message.chat.id, message_id=recognized.message_id)
    else:
        await bot.delete_message(message_id=recognized.message_id, chat_id=recognized.chat.id)
        splited = WhisperRecognition.split_string(result)
        for chunk in splited:
            await message.reply(chunk)
