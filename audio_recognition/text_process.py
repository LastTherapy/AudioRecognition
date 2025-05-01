import re
import nltk
from langfuse.decorators import observe
import nltk
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# Загрузим необходимые ресурсы (раз в проекте):
nltk.download('punkt')          # токенизатор предложений/слов
nltk.download('stopwords')
nltk.download('punkt_tab')
model = SentenceTransformer('all-MiniLM-L6-v2')  # лёгкая и быстрая модель


@observe()
async def ml_split_text(raw_text: str) -> str:
    # 1) Разбиваем на предложения
    sentences = nltk.tokenize.sent_tokenize(raw_text)
    if len(sentences) < 2:
        return raw_text  # слишком короткий текст

    # 2) Получаем эмбеддинги
    emb = model.encode(sentences, convert_to_numpy=True)
    # 3) Считаем «скачки» смысловой дистанции
    dists = [cosine(emb[i], emb[i+1]) for i in range(len(emb)-1)]
    # 4) Авто-порог: среднее + 1σ
    mu, sigma = np.mean(dists), np.std(dists)
    threshold = mu + sigma
    break_idxs = [i for i, d in enumerate(dists) if d > threshold]
    # 5) Формируем параграфы по найденным границам
    paras, start = [], 0
    for idx in break_idxs:
        paras.append(" ".join(sentences[start:idx+1]))
        start = idx+1
    paras.append(" ".join(sentences[start:]))
    # 6) Гарантируем минимум 2 параграфа
    if len(paras) == 1:
        mid = len(sentences) // 2 or 1
        paras = [
            " ".join(sentences[:mid]),
            " ".join(sentences[mid:])
        ]
    # 7) Возвращаем единую строку с двойными переводами строк
    return "\n\n".join(p.strip() for p in paras)

@observe()
async def clean_whisper_text_basic(raw_text: str) -> str:
    # Убираем переносы строк, которые НЕ после точки, вопроса или восклицания
    text = re.sub(r'(?<![\.\?\!])\n', ' ', raw_text)
    # Заменяем множественные пробелы на один
    text = re.sub(r'\s+', ' ', text)
    # Убираем пробелы перед пунктуацией
    text = re.sub(r' \.', '.', text)
    text = re.sub(r' \?', '?', text)
    text = re.sub(r' \!', '!', text)
    return text.strip()
