import logging
import re
from typing import List
from telegram.utils import split_text_for_telegram
from pathlib import Path
from telegram.config import VOICE_STORAGE, VIDEO_STORAGE
#aiogram
from aiogram import Dispatcher
from aiogram.types import Message
from aiogram import F
from aiogram.filters.command import Command
from aiogram.types import CallbackQuery, InlineKeyboardMarkup
# langfuse imports
from langfuse.media import LangfuseMedia
from langfuse.decorators import observe, langfuse_context
# project imports
from audio_recognition.whisper_recogniser import whisper_cli_recognition
from audio_recognition.text_process import clean_whisper_text_basic, ml_split_text
from audio_recognition.utils import convert_voice, extract_audio
from audio_recognition.config import WHISPER_MODEL
from telegram.keyboards import Feedback, get_feedback_keyboard


def setup_handlers(dp: Dispatcher):
    @dp.message(Command("start"))
    async def start_message(message: Message):
        await message.answer("Отправьте голосовое сообщение для распознавания и подождите результата. Либо добавьте в группу и получайте автоматическое распознавнаие видеокружочков и голосовых сообщений.")

    @dp.message(Command("help"))
    async def help_message(message: Message):
        await message.answer("Бот распознает голосовые сообощения. Отправьте ему их лично, либо просто добавьте бота в группу.  Для распознавания используется open-source модель whisper.")

    @dp.message(F.video_note)
    @observe()
    async def video_note_handler(message: Message):
        bot_name = await message.bot.get_my_name(request_timeout=5)
        langfuse_context.update_current_observation(
            name="telegram-video_note",
            metadata={
                "bot_name": bot_name,
                "recognition_model": WHISPER_MODEL,
                "user": message.from_user.username
            }
        )
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
        # tracing languse media
        media = LangfuseMedia(
            file_path=str(wav_path),
            content_type="audio/wav"
        )
        langfuse_context.update_current_observation(input=media)
        text = await whisper_cli_recognition(wav_path)
        langfuse_context.update_current_observation(output=text)
        trace_id: str = langfuse_context.get_current_trace_id()
        keyboard: InlineKeyboardMarkup  = get_feedback_keyboard(trace_id=trace_id)
        if text.strip():
            text = await clean_whisper_text_basic(text)
            text = await ml_split_text(text)
            if len(text) < 4096:
                await message.answer(text)
            else:
                chunks = split_text_for_telegram(text)
                for i, chunk in enumerate(chunks):
                    if i == len(chunks) - 1:
                        # последний кусок — с кнопками
                        await message.answer(chunk, reply_markup=keyboard)
                    else:
                        # промежуточные куски — без кнопок
                        await message.answer(chunk)          
        else:
            logging.info("No text in videonote")
        

    @dp.message(F.voice)
    @observe()
    async def auto_voice_recognition(message: Message):
        bot_name = await message.bot.get_my_name(request_timeout=5)
        langfuse_context.update_current_observation(
            name="telegram-voice",
            metadata={
                "bot_name": bot_name,
                "recognition_model": WHISPER_MODEL,
                "user": message.from_user.username
            }
        )
        trace_id: str = langfuse_context.get_current_trace_id()
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
        wav_path: Path = await convert_voice(destination)
        
        # tracing languse media
        media = LangfuseMedia(
            file_path=str(wav_path),
            content_type="audio/wav"
        )
        langfuse_context.update_current_observation(input=media)
        
        await asnwer_message.edit_text('Успех. Слушаю аудио.')
        # Распознавание через Whisper
        text: str = await whisper_cli_recognition(wav_path)
        
        # await asnwer_message.edit_text('Расставляю абзацы')
        
        langfuse_context.update_current_observation(output=text)
        keyboard: InlineKeyboardMarkup  = get_feedback_keyboard(trace_id=trace_id)
        # Отправляем текст пользователю
        if text.strip():
            text = await clean_whisper_text_basic(text)
            text = await ml_split_text(text)
            if len(text) < 4096:
                await asnwer_message.edit_text(text, reply_markup=keyboard)

            else:
                await asnwer_message.delete()
                chunks = split_text_for_telegram(text)
                for i, chunk in enumerate(chunks):
                    if i == len(chunks) - 1:
                        # последний кусок — с кнопками
                        await message.answer(chunk, reply_markup=keyboard)
                    else:
                        # промежуточные куски — без кнопок
                        await message.answer(chunk)
                    
        else:
            await asnwer_message.edit_text("Аудио без слов")
            


    @dp.message()
    async def log_text_messages(message: Message):
        logging.info(f"{message.chat.full_name}: {message.text}")
        
    
    @dp.callback_query(Feedback.filter())
    @observe()
    async def feedback_handler(query: CallbackQuery, callback_data: Feedback):
        print(f"feedback pressed {callback_data.val}")
        pass



