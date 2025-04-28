import os
from dotenv import load_dotenv
# Путь к директории скрипта
load_dotenv()

script_dir = os.path.dirname(os.path.abspath(__file__))
# Путь к папке voice относительно директории скрипта
VOICE_STORAGE = os.path.join(script_dir, 'voice/')
VIDEO_STORAGE = os.path.join(script_dir, 'video/')

WHISPER_CLI = os.getenv("WHISPER_CLI")
WHISPER_MODEL = os.getenv("WHISPER_MODEL")
TG_TOKEN = os.getenv("TG_TOKEN")


media_autoremove = False
