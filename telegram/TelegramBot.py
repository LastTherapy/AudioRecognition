import asyncio
import logging
import sys
from os import getenv
from aiogram import Bot, Dispatcher, Router, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram.types import PhotoSize
from aiogram.utils.markdown import hbold
from aiogram.utils.keyboard import InlineKeyboardBuilder
from settings import TOKEN, VOICE_SRORAGE
from aiogram import F
from aiogram.types import FSInputFile, ContentType, ReactionTypeEmoji, CallbackQuery
import WhisperRecognition
from aiogram.client.default import DefaultBotProperties


bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp: Dispatcher = Dispatcher()


@dp.message(F.content_type.in_({'voice'}))
async def auto_voice_recognition(message: Message):
        logging.info('voice received from ' + message.from_user.full_name + ' with id ' + str(message.from_user.id))
        await voice_recognition(message)

async def voice_recognition(message: Message, model: str = 'small'):
    voice_id = message.voice.file_id
    logging.debug('voice id: ' + voice_id)
    file = await bot.get_file(voice_id)
    logging.debug('file: ' + str(file))
    file_path = file.file_path
    logging.debug('file path: ' + file_path)
    destination_file = f'{VOICE_SRORAGE}{message.message_id}.oga'
    logging.debug('destination file: ' + destination_file)
    recognized: Message = await message.reply("Скачивайю файл для распознавания...")
    await bot.download_file(file_path, destination=destination_file)
    print("voice downloaded")
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    await bot.edit_message_text("Начинаю распознавание... Использую самую мощную модель, это может занять время...", chat_id=message.chat.id, message_id=recognized.message_id)
    try:
        result = WhisperRecognition.recognition(destination_file, model)
    except Exception():
        logging.exception("Error in voice recognition")
        result = "Sorry, no more GPU memory available just now. Error(("
    if len(result) == 0:
        result = "Sorry, no text in voice recognition."
    if len(result) < 4096:
        await bot.edit_message_text(result, chat_id=message.chat.id, message_id=recognized.message_id)
    else:
        await bot.delete_message(message_id=recognized.message_id, chat_id=recognized.chat.id)
        splited = WhisperRecognition.split_string(result)
        for chunk in splited:
            print(chunk)
            await message.reply(chunk)

async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    asyncio.run(main())