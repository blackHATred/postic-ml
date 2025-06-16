"""Главный файл приложения."""
import uvicorn
from fastapi import FastAPI, Request

from config.settings import HOST, PORT
from core.init import init_clients, init_models
from api.publication import get_publication
from api.comments import (
    get_sentiment, get_ticket_synt, get_ticket_llm, 
    get_summary, fix_text, answer_comment
)

# Инициализация клиентов и моделей
CLIENT, REDIS_CLIENT, SEARCHER = init_clients()
SENTIMENT_MODEL, SEQ_MODEL, TICKET_MODEL = init_models()

app = FastAPI()


@app.get("/publication")
async def publication_endpoint(request: Request):
    """Эндпоинт для генерации публикаций."""
    return await get_publication(request, CLIENT, REDIS_CLIENT, SEARCHER)


@app.get("/sentiment")
async def sentiment_endpoint(request: Request):
    """Эндпоинт для анализа тональности."""
    return await get_sentiment(request, SENTIMENT_MODEL)


@app.get("/ticket_synt")
async def ticket_synt_endpoint(request: Request):
    """Эндпоинт для определения необходимости поддержки (ML)."""
    return await get_ticket_synt(request, SEQ_MODEL, TICKET_MODEL)


@app.get("/ticket_llm")
async def ticket_llm_endpoint(request: Request):
    """Эндпоинт для определения необходимости поддержки (LLM)."""
    return await get_ticket_llm(request)


@app.get("/sum")
async def summary_endpoint(request: Request):
    """Эндпоинт для суммаризации комментариев."""
    return await get_summary(request)


@app.get("/fix")
async def fix_endpoint(request: Request):
    """Эндпоинт для исправления текста."""
    return await fix_text(request)


@app.get("/ans")
async def answer_endpoint(request: Request):
    """Эндпоинт для генерации ответов на комментарии."""
    return await answer_comment(request)


if __name__ == "__main__":
    print("Сервис работает.")
    uvicorn.run(app, host=HOST, port=PORT)
