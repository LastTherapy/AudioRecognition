import asyncio
import logging
import os, sys
import gc
from aiogram import Bot, Dispatcher
from handlers import setup_handlers
from dotenv import load_dotenv



async def main() -> None:
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    load_dotenv()
    bot = Bot(os.getenv("TG_TOKEN"))
    dp: Dispatcher = Dispatcher()
    setup_handlers(dp)
    await dp.start_polling(bot)

if __name__ == "__main__":
    gc.enable()
    gc.collect()
    asyncio.run(main())
