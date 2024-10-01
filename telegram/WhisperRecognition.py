import whisper
import multiprocessing as mp



def isolated_recognition(destination_file: str, queue, model: str = "turbo"):
    model = whisper.load_model(model)
    result = model.transcribe(destination_file)
    queue.put(result['text'])


def recognition(destination_file: str, model: str = "turbo"):
    model = "turbo" if model not in ['tiny', 'base', 'small', 'medium', 'large', 'turbo'] else model
    print(f"using {model} model")

    queue = mp.Queue()
    process = mp.Process(target=isolated_recognition, args=(destination_file, queue, model))
    process.start()
    process.join()
    result = queue.get()
    return result

def split_string(s, chunk_size=4096):
    for start in range(0, len(s), chunk_size):
        yield s[start:start + chunk_size]

