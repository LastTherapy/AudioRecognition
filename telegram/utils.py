import os
import logging
from time import sleep

from aiogram import Bot
from aiogram.types import Message
from config import VOICE_STORAGE
import WhisperRecognition
from moviepy import VideoFileClip
from config import VIDEO_STORAGE


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
    audio_path = os.path.join(VOICE_STORAGE, f'{message.message_id}.wav')
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    return audio_path



async def perform_voice_recognition(message: Message, model: str = 'small', audio_path = None):
    bot = message.bot
    if audio_path is None:
        destination_file = await download_file(bot, message)
    else:
        destination_file = audio_path

    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    recognized: Message = await message.reply(f"Начинаю распознавание... Использую {model} модель...")

    try:
        result = WhisperRecognition.recognition(destination_file, model)['result']
    except RuntimeError as e:
        logging.exception("Error in voice recognition")
        result = "Sorry, no more GPU memory available just now. Try again later."
        await message.reply(result)
        sleep(5)
        await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)
        return

    if not result:
        result = "Sorry, no text in voice recognition."

    if len(result) < 4096:
        await bot.edit_message_text(result, chat_id=message.chat.id, message_id=recognized.message_id)
    else:
        await bot.delete_message(message_id=recognized.message_id, chat_id=recognized.chat.id)
        split = WhisperRecognition.split_string(result)
        for chunk in split:
            await message.reply(chunk)

