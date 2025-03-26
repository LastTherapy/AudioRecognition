import asyncio
import logging
import os, sys

import uvicorn
from aiogram import Bot, Dispatcher
from dotenv import load_dotenv
from telegram.handlers import setup_handlers
from api_server import app

async def run_bot() -> None:
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    load_dotenv()

    bot = Bot(os.getenv("TG_TOKEN"))
    dp: Dispatcher = Dispatcher()
    setup_handlers(dp)
    await dp.start_polling(bot)


async def run_api():
    config = uvicorn.Config(app, host="127.0.0.1", port=8555, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    await asyncio.gather(
        run_bot(),
        run_api()
    )

if __name__ == "__main__":
    asyncio.run(main())


