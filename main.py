# Запуск: uvicorn main:app --reload

import os
import logging
from contextlib import asynccontextmanager
from typing import List
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from openrouter import OpenRouter
from dotenv import load_dotenv

# Загрузка переменных из .env
load_dotenv()

# Логи
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Конфигурация (секреты только через переменные окружения)-
API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
if not API_KEY:
    raise ValueError("Переменная окружения OPENROUTER_API_KEY не задана! Создайте файл .env")

EMBEDDER_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "z-ai/glm-4.5-air:free"
DOCS_FOLDER = "docs"                       # папка с PDF
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Глобальные объекты (заполняются при старте)
embedder = None
index = None
documents = []      # тексты чанков
metadatas = []      # [{"file": ..., "chunk_index": ...}]

# Вспомогательные функции (из Colab)
def extract_text_from_pdf(pdf_path):
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_index():
    global embedder, index, documents, metadatas
    logger.info("Загрузка модели эмбеддингов...")
    embedder = SentenceTransformer(EMBEDDER_NAME)

    logger.info("Чтение PDF из папки %s...", DOCS_FOLDER)
    docs_texts = {}
    for fn in os.listdir(DOCS_FOLDER):
        if fn.endswith(".pdf"):
            path = os.path.join(DOCS_FOLDER, fn)
            docs_texts[fn] = extract_text_from_pdf(path)
            logger.info("  %s: %d символов", fn, len(docs_texts[fn]))

    documents = []
    metadatas = []
    for fn, text in docs_texts.items():
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"file": fn, "chunk_index": i})

    logger.info("Всего чанков: %d", len(documents))

    logger.info("Создание эмбеддингов...")
    embeddings = embedder.encode(documents, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')

    logger.info("Построение FAISS-индекса...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    logger.info("Индекс готов. Размерность: %d, векторов: %d", dimension, index.ntotal)

def retrieve(query: str, top_k: int = 5):
    query_emb = embedder.encode([query]).astype('float32')
    distances, indices = index.search(query_emb, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "doc_id": metadatas[idx]['file'],
            "snippet": documents[idx][:300],
            "score": float(dist),
            "full_text": documents[idx]
        })
    return results

def build_prompt(query: str, retrieved: list):
    context_parts = []
    for i, item in enumerate(retrieved):
        context_parts.append(f"[Фрагмент {i+1}] (файл: {item['doc_id']})\n{item['full_text']}")
    context = "\n\n".join(context_parts)

    system_msg = (
        "Ты – ассистент, отвечающий строго по предоставленному контексту.\n"
        "1. Прочитай контекст.\n"
        "2. Найди факты, относящиеся к вопросу.\n"
        "3. Дай краткий ответ, используя только эти факты.\n"
        "4. Если информации недостаточно, напиши: 'Информации в предоставленных документах недостаточно.'\n"
        "Не добавляй ничего от себя."
    )
    prompt = f"{system_msg}\n\nКонтекст:\n{context}\n\nВопрос: {query}\nОтвет:"
    return prompt

def ask_llm(question: str):
    retrieved = retrieve(question, top_k=5)
    prompt = build_prompt(question, retrieved)

    logger.info("Отправка запроса к OpenRouter...")
    try:
        with OpenRouter(api_key=API_KEY) as client:
            response = client.chat.send(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.2,
            )
        answer = response.choices[0].message.content
    except Exception as e:
        logger.error("Ошибка вызова LLM: %s", str(e))
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    sources = []
    for item in retrieved:
        sources.append({
            "doc_id": item['doc_id'],
            "score": item['score'],
            "snippet": item['snippet']
        })
    return {"answer": answer, "sources": sources}

# Pydantic модели
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, description="Вопрос пользователя")

class AskResponse(BaseModel):
    answer: str
    sources: List[dict]

# FastAPI приложение
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Инициализация индекса при старте сервиса...")
    build_index()
    yield
    logger.info("Сервис остановлен.")

app = FastAPI(
    title="RAG LLM Service",
    description="Сервис для ответов на вопросы по учебным документам (Linux, Docker, Windows)",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_ready": embedder is not None and index is not None,
        "documents_loaded": len(documents),
        "index_size": index.ntotal if index else 0,
    }

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    logger.info("Получен вопрос: %s", req.question)
    if not embedder or not index:
        raise HTTPException(status_code=503, detail="Индекс ещё не готов. Попробуйте позже.")
    if len(req.question) > 500:
        raise HTTPException(status_code=400, detail="Вопрос слишком длинный (макс. 500 символов)")
    result = ask_llm(req.question)
    logger.info("Ответ отправлен, источников: %d", len(result["sources"]))
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
