import settings
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
import torch
import tempfile
import time
from pydub import AudioSegment
import os
import numpy as np
# --- Добавьте эти импорты для отладки NumPy ---
from scipy.io.wavfile import write as write_wav
import logging # Для лучшего логирования
# ---------------------------------------------

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()

# Разрешённые форматы
ALLOWED_EXTENSIONS = {".wav", ".oga", ".mp3", ".flac", ".m4a"}
MAX_FILE_SIZE_MB = 256

# --- Глобальные переменные ---
MODEL_NAME = "openai/whisper-large-v3" # Используем базовую модель для теста
TORCH_DTYPE = torch.float32

logger.info("🚀 Загружаем модель и процессор...")
try:
    # Загружаем сразу на CPU
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to("cpu") # Явно на CPU при старте

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    logger.info("✅ Модель и процессор готовы.")
except Exception as e:
    logger.exception("❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить модель или процессор!")
    # В реальном приложении здесь можно завершить работу или перейти в аварийный режим
    model = None
    processor = None

# --- Пайплайн лучше создавать позже, когда известно устройство ---

def is_gpu_available(threshold_mb: int = 10000) -> bool:
    """Проверяет, есть ли достаточно свободной памяти на GPU."""
    if model is None: # Если модель не загрузилась, GPU использовать не будем
        return False
    if torch.cuda.is_available():
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_mem_mb = free_mem / (1024 ** 2)
            logger.info(f"ℹ️ Доступно памяти GPU: {free_mem_mb:.2f} MB")
            return free_mem_mb > threshold_mb
        except Exception as e:
            logger.warning(f"⚠️ Ошибка при проверке памяти GPU: {e}")
            return False
    return False


@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    # --- ЯВНО УКАЗЫВАЕМ ИСПОЛЬЗОВАНИЕ ГЛОБАЛЬНЫХ ПЕРЕМЕННЫХ ---
    global model, processor
    # --------------------------------------------------------

    if model is None or processor is None:
         raise HTTPException(status_code=503, detail="Сервис временно недоступен (модель не загружена).")

    tmp_input_path = None # Инициализируем для finally
    try:
        # Проверка расширения
        ext = os.path.splitext(file.filename)[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Неподдерживаемый формат файла: {ext}")

        contents = await file.read()
        size_mb = len(contents) / (1024 * 1024)
        logger.info(f"Получен файл: {file.filename}, Размер: {size_mb:.2f} MB, Тип: {file.content_type}")

        # --- ОТЛАДКА: Сохраняем полученный файл ---
        debug_dir = "debug_audio"
        os.makedirs(debug_dir, exist_ok=True)
        debug_filename = os.path.join(debug_dir, f"received_{time.time()}_{file.filename}")
        try:
            with open(debug_filename, "wb") as df:
                df.write(contents)
            logger.info(f"📝 Исходный файл сохранен для отладки: {debug_filename}")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось сохранить отладочный файл: {e}")
        # --- КОНЕЦ ОТЛАДКИ ---

        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(status_code=413, detail=f"Файл слишком большой ({size_mb:.2f} МБ). Максимум — {MAX_FILE_SIZE_MB} МБ") # 413 Payload Too Large

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_input:
            tmp_input.write(contents)
            tmp_input_path = tmp_input.name
            logger.info(f"📄 Временный файл сохранен: {tmp_input_path}")

        samples = None
        duration_sec = 0
        # --- Загрузка и предобработка аудио ---
        try:
            logger.info(f"🎧 Загрузка и обработка аудио: {tmp_input_path}")
            audio = AudioSegment.from_file(tmp_input_path)
            duration_sec = len(audio) / 1000.0

            audio = audio.set_frame_rate(16000).set_channels(1)

            # Нормализация к float32 [-1, 1]
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            # Проверяем sample_width для корректной нормализации
            if audio.sample_width == 2: # 16 бит
                samples /= 32768.0
            elif audio.sample_width == 4: # 32 бита
                 # Максимальное значение для float32 при чтении из int32
                samples /= 2147483648.0
            elif audio.sample_width == 1: # 8 бит
                 samples /= 128.0 # Центральное значение 0
            else:
                 logger.warning(f"Нестандартная ширина сэмпла: {audio.sample_width}. Используется общая нормализация.")
                 # Общая нормализация, может быть неточной
                 samples /= np.iinfo(f'int{audio.sample_width * 8}').max

            samples = np.clip(samples, -1.0, 1.0) # Убедимся, что в диапазоне

            logger.info(f"✅ Аудио обработано: длительность {duration_sec:.2f} сек, shape={samples.shape}, dtype={samples.dtype}")

            # --- ОТЛАДКА: Сохраняем NumPy массив как WAV ---
            debug_numpy_filename = os.path.join(debug_dir, f"numpy_{time.time()}_{file.filename}.wav")
            try:
                # Нормализуем в int16 для сохранения как стандартный WAV
                samples_int16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
                write_wav(debug_numpy_filename, 16000, samples_int16) # 16000 Hz
                logger.info(f"📝 NumPy массив сохранен для отладки: {debug_numpy_filename}")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось сохранить отладочный NumPy WAV: {e}")
            # --- КОНЕЦ ОТЛАДКИ ---

        except Exception as e:
            logger.exception(f"❌ Ошибка при обработке аудиофайла {file.filename} с помощью pydub: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка обработки аудиофайла: {e}")
        # --- Конец обработки аудио ---

        if samples is None:
             raise HTTPException(status_code=500, detail="Не удалось получить аудиоданные.")

        # Определяем устройство
        device_used = "cpu"
        current_device = torch.device("cpu")
        dtype_used = TORCH_DTYPE
        model_on_gpu = False # Флаг, чтобы знать, нужно ли перемещать обратно

        if is_gpu_available():
            logger.info("🚀 Обнаружен GPU с достаточной памятью. Попытка использовать GPU...")
            try:
                model.to("cuda") # Используем глобальную модель
                current_device = torch.device("cuda")
                device_used = "cuda"
                model_on_gpu = True
                # dtype_used = torch.float16 # Пока не используем float16 для простоты
                logger.info("✅ Модель перемещена на GPU.")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось переместить модель на GPU: {e}. Используем CPU.")
                # Убедимся, что модель точно на CPU, если она уже там - ошибки не будет
                model.to("cpu") # Используем глобальную модель
                current_device = torch.device("cpu")
                device_used = "cpu"
                dtype_used = TORCH_DTYPE
                model_on_gpu = False
        else:
            logger.info("ℹ️ GPU недоступен или недостаточно памяти. Используем CPU.")
            model.to("cpu") # Используем глобальную модель (на всякий случай)
            current_device = torch.device("cpu")
            device_used = "cpu"
            model_on_gpu = False


        # ⚙️ Создаём pipeline
        logger.info(f"⚙️ Создание пайплайна для устройства: {device_used}")
        asr_pipeline_local = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            return_timestamps=True,
            device=current_device,
            torch_dtype=dtype_used
        )
        logger.info(f"✅ Пайплайн создан для {device_used}.")

        # 🕒 Время и запуск распознавания
        start_time = time.time()
        logger.info("🎤 Запуск распознавания...")

        # Попробуем понизить порог VAD
        generate_kwargs = {"no_speech_threshold": 0.5} # Экспериментируем

        result = asr_pipeline_local(samples.copy())
        elapsed = time.time() - start_time

        logger.info(f"⏱️ Распознавание завершено за {elapsed:.2f} сек на устройстве: {device_used}")
        logger.info(f"🎧 Полный результат: {result}") # Логируем полный результат

        # Собираем текст из чанков (так как return_timestamps='word')
        final_text = " ".join([chunk['text'] for chunk in result.get("chunks", [])]).strip()

        logger.info(f"📝 Результат (текст): {final_text[:200]}...")

        return {
                "filename": file.filename,
                "format": ext,
                "duration_seconds": duration_sec,
                "text": final_text, # Возвращаем собранный текст
                "device": device_used,
                "inference_time": elapsed,
                "full_result": result # Можно вернуть для отладки на клиенте
            }

    except HTTPException as e:
        # Логируем HTTP исключения тоже
        logger.error(f"HTTP ошибка: {e.status_code} - {e.detail}")
        raise e
    except Exception as e:
        logger.exception(f"❌ Непредвиденная ошибка при обработке запроса для {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")
    finally:
        # 🔁 Перевод обратно на CPU, если модель перемещалась на GPU
        if model_on_gpu:
            try:
                logger.info("🔁 Перемещаем модель обратно на CPU...")
                model.to("cpu")
                torch.cuda.empty_cache()
                logger.info("✅ Модель на CPU, кэш GPU очищен.")
            except Exception as e:
                 logger.error(f"⚠️ Ошибка при перемещении модели на CPU или очистке кэша: {e}")

        # --- Очистка временного файла ---
        if tmp_input_path and os.path.exists(tmp_input_path):
            try:
                os.remove(tmp_input_path)
                logger.info(f"🧹 Временный файл удален: {tmp_input_path}")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось удалить временный файл {tmp_input_path}: {e}")


if __name__ == "__main__":
    port = getattr(settings, 'api_port', 8000)
    logger.info(f"🚀 Запуск сервера на порту {port}")
    # Запускаем через uvicorn для поддержки async
    # Используйте reload=True только для разработки
    uvicorn.run("your_module_name:app", host="0.0.0.0", port=port, reload=False)
    # Замените "your_module_name" на имя вашего Python файла (без .py)