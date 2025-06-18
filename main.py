"""Главный файл приложения."""
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config.settings import HOST, PORT
from core.init import init_clients, init_models, preload_ollama_models
from api.publication import get_publication
from api.comments import (
    get_sentiment, get_ticket_synt, get_ticket_llm, 
    get_summary, fix_text, answer_comment
)
from models.requests import (
    PublicationRequest, SentimentRequest, TicketRequest,
    SummaryRequest, FixTextRequest, AnswerCommentRequest
)
from models.responses import (
    PublicationResponse, SentimentResponse, TicketResponse,
    SummaryResponse, FixTextResponse, AnswerCommentResponse, ErrorResponse
)

# Инициализация клиентов и моделей
CLIENT, REDIS_CLIENT, SEARCHER = init_clients()
SENTIMENT_MODEL, SEQ_MODEL, TICKET_MODEL = init_models()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    # Startup
    print("🚀 Запуск приложения...")
    await preload_ollama_models()
    print("✅ Приложение готово к работе")
    
    yield
    
    # Shutdown
    print("🔄 Завершение работы приложения...")
    # Здесь можно добавить cleanup логику если нужно
    print("👋 Приложение завершено")


app = FastAPI(
    lifespan=lifespan,
    title="Postic ML API",
    description="""
    🤖 **AI-Powered Social Media Management API**
    
    Мощный API для автоматизации управления контентом в социальных сетях с использованием искусственного интеллекта.
    
    ## Основные возможности:
    
    - 📝 **Генерация контента** с использованием LLM и поиска
    - 😊 **Анализ тональности** комментариев  
    - 🎫 **Определение тикетов поддержки** с ML и LLM
    - 📊 **Суммаризация комментариев** к постам
    - ✏️ **Исправление орфографии** и пунктуации
    - 💬 **Генерация ответов** на комментарии
    
    ## Документация:
    
    - 📖 [Примеры использования](https://github.com/your-repo/API_EXAMPLES.md)
    - 🔄 [Руководство по миграции](https://github.com/your-repo/MIGRATION.md)
    """,
    version="1.0.0",
    tags_metadata=[
        {
            "name": "content",
            "description": "Операции с контентом и публикациями",
        },
        {
            "name": "sentiment", 
            "description": "Анализ тональности и эмоций",
        },
        {
            "name": "support",
            "description": "Определение необходимости поддержки",
        },
        {
            "name": "text",
            "description": "Обработка и исправление текста",
        },
        {
            "name": "comments",
            "description": "Работа с комментариями",
        },
        {
            "name": "health",
            "description": "Системные эндпоинты",
        },
    ]
)


# Настройка CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "https://postic.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/publication", response_model=PublicationResponse, responses={400: {"model": ErrorResponse}}, tags=["content"])
async def publication_endpoint(request: PublicationRequest):
    """Эндпоинт для генерации публикаций."""
    return await get_publication(request, CLIENT, REDIS_CLIENT, SEARCHER)


@app.post("/publication/stream", tags=["content"])
async def publication_stream_endpoint(request: PublicationRequest):
    """Эндпоинт для потоковой генерации публикаций через SSE."""
    from api.publication import get_publication_stream
    
    async def event_generator():
        async for data in get_publication_stream(request, CLIENT, REDIS_CLIENT, SEARCHER):
            yield data
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )


@app.post("/sentiment", response_model=SentimentResponse, responses={400: {"model": ErrorResponse}}, tags=["sentiment"])
async def sentiment_endpoint(request: SentimentRequest):
    """Эндпоинт для анализа тональности."""
    return await get_sentiment(request, SENTIMENT_MODEL)


@app.post("/ticket_synt", response_model=TicketResponse, responses={400: {"model": ErrorResponse}}, tags=["support"])
async def ticket_synt_endpoint(request: TicketRequest):
    """Эндпоинт для определения необходимости поддержки (ML)."""
    return await get_ticket_synt(request, SEQ_MODEL, TICKET_MODEL)


@app.post("/ticket_llm", response_model=TicketResponse, responses={400: {"model": ErrorResponse}}, tags=["support"])
async def ticket_llm_endpoint(request: TicketRequest):
    """Эндпоинт для определения необходимости поддержки (LLM)."""
    return await get_ticket_llm(request)


@app.post("/sum", response_model=SummaryResponse, responses={400: {"model": ErrorResponse}}, tags=["comments"])
async def summary_endpoint(request: SummaryRequest):
    """Эндпоинт для суммаризации комментариев."""
    return await get_summary(request)


@app.post("/fix", response_model=FixTextResponse, responses={400: {"model": ErrorResponse}}, tags=["text"])
async def fix_endpoint(request: FixTextRequest):
    """Эндпоинт для исправления текста."""
    return await fix_text(request)


@app.post("/ans", response_model=AnswerCommentResponse, responses={400: {"model": ErrorResponse}}, tags=["comments"])
async def answer_endpoint(request: AnswerCommentRequest):
    """Эндпоинт для генерации ответов на комментарии."""
    return await answer_comment(request)


@app.get("/health", tags=["health"])
async def health_check():
    """Проверка работоспособности сервиса."""
    return {"status": "healthy", "message": "Сервис работает"}


@app.get("/sentiment", response_model=SentimentResponse, responses={400: {"model": ErrorResponse}}, tags=["sentiment"])
async def sentiment_get_endpoint(comment: str):
    """GET эндпоинт для анализа тональности с query параметром."""
    request = SentimentRequest(comment=comment)
    return await get_sentiment(request, SENTIMENT_MODEL)


@app.get("/ticket_synt", response_model=TicketResponse, responses={400: {"model": ErrorResponse}}, tags=["support"])
async def ticket_synt_get_endpoint(comment: str):
    """GET эндпоинт для определения необходимости поддержки (ML) с query параметром."""
    request = TicketRequest(comment=comment)
    return await get_ticket_synt(request, SEQ_MODEL, TICKET_MODEL)


@app.get("/ticket_llm", response_model=TicketResponse, responses={400: {"model": ErrorResponse}}, tags=["support"])
async def ticket_llm_get_endpoint(comment: str):
    """GET эндпоинт для определения необходимости поддержки (LLM) с query параметром."""
    request = TicketRequest(comment=comment)
    return await get_ticket_llm(request)


@app.get("/health/models", tags=["health"])
async def models_status():
    """Проверка статуса загруженных моделей Ollama."""
    from services.llm_service import check_model_status
    return await check_model_status()


@app.get("/health", tags=["health"])
async def health_check():
    """Базовая проверка здоровья системы."""
    return {
        "status": "healthy",
        "message": "Postic ML API is running",
        "components": {
            "qdrant": "connected",
            "redis": "connected", 
            "ollama": "available"
        }
    }


# Подключение статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    print("Сервис работает.")
    uvicorn.run(app, host=HOST, port=PORT)
