import asyncio
import logging
import sys
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from config import TOKEN
from handlers import setup_handlers

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

# Создание экземпляра бота и диспетчера
bot = Bot(token=TOKEN)
dp: Dispatcher = Dispatcher()

async def main() -> None:
    # Регистрация обработчиков
    setup_handlers(dp)
    # Запуск бота
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
