import asyncio
import aiohttp
import logging
import os
from time import sleep
import mimetypes
import requests
from aiogram import Bot
from aiogram.types import Message
from moviepy import VideoFileClip

from settings import api_url
from telegram import WhisperRecognition
from telegram.config import VOICE_STORAGE, VIDEO_STORAGE, media_autoremove








# async def perform_voice_recognition(message: Message, model: str = 'small', audio_path=None):
#     bot = message.bot
#     if audio_path is None:
#         destination_file = await download_file(bot, message)
#     else:
#         destination_file = audio_path
#
#     await bot.send_chat_action(chat_id=message.chat.id, action="typing")
#     recognized: Message = await message.reply(f"Начинаю распознавание...")
#
#     try:
#         result: str = WhisperRecognition.recognition(destination_file, model)['result']
#         if not result:
#             return
#         # using llm for correct mistakes
#         await recognized.edit_text(f"Отправляю результат на корректировку...")
#         result = await improve_recognition(result)
#
#     except Exception as e:
#         logging.exception("Error in voice recognition")
#         print(e)
#         result = "Sorry, error in voice recognition. Try again later."
#         await message.reply(result)
#         sleep(5)
#         await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)
#         return
#
#     if len(result) < 4096:
#         if len(result) > 0:
#             await recognized.edit_text(result)
#         # await bot.edit_message_text(result, chat_id=message.chat.id, message_id=recognized.message_id)
#         else:
#             await recognized.edit_text('В медиа нет текста')
#             sleep(5)
#             await recognized.delete()
#     else:
#         await recognized.delete()
#         split = WhisperRecognition.split_string(result)
#         for chunk in split:
#             await message.reply(chunk)
#
#     if media_autoremove:
#         os.remove(destination_file)

TRANSCRIBE_URL = "http://localhost:8555/transcribe/"

async def download_file(bot: Bot, message: Message) -> str:
    file = await bot.get_file(message.voice.file_id)
    destination_file = os.path.join(VOICE_STORAGE, f'{message.message_id}.oga')
    await bot.download_file(file.file_path, destination_file)
    return destination_file


async def download_video_circle(bot: Bot, message: Message) -> str:
    file = await bot.get_file(message.video_note.file_id)
    destination_file = os.path.join(VIDEO_STORAGE, f'{message.message_id}.mp4')
    await bot.download_file(file.file_path, destination_file)
    return destination_file


async def extract_audio(message: Message) -> str:
    bot = message.bot
    file: str = await download_video_circle(bot, message)
    video = VideoFileClip(file)
    if media_autoremove:
        os.remove(file)
    audio_path = os.path.join(VOICE_STORAGE, f'{message.message_id}.wav')
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    return audio_path


async def perform_voice_recognition(message: Message, audio_path=None):
    bot = message.bot

    if audio_path is None:
        destination_file = await download_file(bot, message)
    else:
        destination_file = audio_path

    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    recognized: Message = await message.reply(f"Провожу распознавание...")

    try:
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            with open(destination_file, "rb") as f:
                # Определяем Content-Type на основе расширения файла
                content_type, _ = mimetypes.guess_type(destination_file)
                if content_type is None:
                    # Устанавливаем тип по умолчанию или определяем по расширению вручную
                    if destination_file.lower().endswith('.oga'):
                        content_type = 'audio/ogg'
                    elif destination_file.lower().endswith('.wav'):
                        content_type = 'audio/wav'
                    # Добавьте другие типы при необходимости
                    else:
                        content_type = 'application/octet-stream'  # Общий тип

                data = aiohttp.FormData()
                data.add_field(
                    "file",
                    f,
                    filename=os.path.basename(destination_file),
                    # Используем определенный content_type
                    content_type=content_type
                )
                logging.info(f"Отправка файла {destination_file} с Content-Type: {content_type}")  # Логирование

                async with session.post(TRANSCRIBE_URL, data=data) as response:
                        if response.status != 200:
                            raise Exception(f"Ошибка от сервера: {response.status}")
                        result_json = await response.json()
                        result = result_json.get("text", "")


        if not result:
            await recognized.edit_text("В медиа не найдено текста.")
            sleep(5)
            await recognized.delete()
            return


        # await recognized.edit_text("Отправляю результат на корректировку...")
        # result = await improve_recognition(result)

    except Exception as e:
        logging.exception("Error in voice recognition")
        await message.reply("Произошла ошибка при распознавании. Попробуйте позже.")
        sleep(5)
        await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)
        return

    # Отправка результата
    if len(result) < 4096:
        await recognized.edit_text(result)
    else:
        await recognized.delete()
        for chunk in WhisperRecognition.split_string(result):
            await message.reply(chunk)

    if media_autoremove:
        os.remove(destination_file)


async def improve_recognition(data: str):
    post_data = {
        "message": data
    }
    response = requests.post(api_url, json=post_data)
    print(response)
    print(response.text)
    if response.status_code == 200:
        return response.text
    else:
        return data

