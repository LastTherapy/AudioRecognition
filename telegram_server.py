import asyncio
import logging
import os, sys
import uvicorn
from aiogram import Bot, Dispatcher
from dotenv import load_dotenv
from telegram.handlers import setup_handlers
from telegram.config import TG_TOKEN

async def run_bot() -> None:
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    
    bot = Bot(TG_TOKEN)
    dp: Dispatcher = Dispatcher()
    setup_handlers(dp)
    await dp.start_polling(bot)

async def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    await asyncio.gather(
        run_bot(),
    )

if __name__ == "__main__":
    asyncio.run(main())


