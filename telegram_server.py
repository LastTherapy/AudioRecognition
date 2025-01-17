import asyncio
import logging
import os, sys
from aiogram import Bot, Dispatcher
from dotenv import load_dotenv
from telegram.handlers import setup_handlers


async def main() -> None:
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    load_dotenv()

    bot = Bot(os.getenv("TG_TOKEN"))
    dp: Dispatcher = Dispatcher()
    setup_handlers(dp)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
