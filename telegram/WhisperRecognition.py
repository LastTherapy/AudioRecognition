import torch.cuda
import whisper
import multiprocessing as mp
import pynvml

# a dictionary for memory checking
whisper_models_vram = {
    "tiny": 1000,    # ~1 ГБ
    "base": 1000,    # ~1 ГБ
    "small": 2000,   # ~2 ГБ
    "medium": 5000,  # ~5 ГБ
    "large": 10000   # ~10 ГБ
}



# Словарь с требованиями к видеопамяти (в мегабайтах) для каждой модели Whisper
whisper_models_vram = {
    "tiny": 1000,    # ~1 ГБ
    "base": 1000,    # ~1 ГБ
    "small": 2000,   # ~2 ГБ
    "medium": 5000,  # ~5 ГБ
    "large": 10000   # ~10 ГБ
}


def check_and_select_device(model: str):
    """
    Проверяет наличие достаточного количества свободной видеопамяти для модели
    и возвращает "cuda:X" для подходящего GPU или "cpu", если памяти недостаточно.
    """
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    required_memory = whisper_models_vram.get(model, 0)  # Получаем требования модели

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory_mb = mem_info.free / 1024**2

        print(f"GPU {i}:")
        print(f"  Общая память: {mem_info.total / 1024**2:.2f} MB")
        print(f"  Использованная память: {mem_info.used / 1024**2:.2f} MB")
        print(f"  Свободная память: {free_memory_mb:.2f} MB")

        if free_memory_mb >= required_memory:
            pynvml.nvmlShutdown()
            return f"cuda:{i}"  # Возвращаем идентификатор подходящего GPU

    pynvml.nvmlShutdown()
    return "cpu"  # Если памяти недостаточно, возвращаем CPU


def isolated_recognition(destination_file: str, queue, model: str = "turbo"):
    device=check_and_select_device(model)
    print(f"Using {device}")
    try:
        model = whisper.load_model(model, device=device)
        result = model.transcribe(destination_file)
        queue.put({'result': result['text']})
    except torch.cuda.OutOfMemoryError:
        # we can't handle it via isolated process, so just put a message
        queue.put({'error': "Out of memory"})


def recognition(destination_file: str, model: str = "turbo"):
    model = "turbo" if model not in ['tiny', 'base', 'small', 'medium', 'large', 'turbo'] else model
    print(f"using {model} model")

    queue = mp.Queue()
    process = mp.Process(target=isolated_recognition, args=(destination_file, queue, model))
    process.start()
    process.join()
    result = queue.get()
    if "error" in result:
        raise RuntimeError(result['error'])
    return result

def split_string(s, chunk_size=4096):
    for start in range(0, len(s), chunk_size):
        yield s[start:start + chunk_size]

