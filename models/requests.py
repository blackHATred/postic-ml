"""Модели запросов для API эндпоинтов."""
from typing import List, Optional
from pydantic import BaseModel, Field


class PublicationRequest(BaseModel):
    """Запрос для генерации публикации."""
    query: str = Field(..., description="Поисковый запрос для генерации публикации")


class SentimentRequest(BaseModel):
    """Запрос для анализа тональности."""
    comment: str = Field(..., description="Комментарий для анализа тональности")


class TicketRequest(BaseModel):
    """Запрос для определения необходимости поддержки."""
    comment: str = Field(..., description="Комментарий для анализа")


class SummaryRequest(BaseModel):
    """Запрос для суммаризации комментариев."""
    comments: str = Field(..., description="Комментарии для суммаризации")


class FixTextRequest(BaseModel):
    """Запрос для исправления текста."""
    text: str = Field(..., description="Текст для исправления")


class AnswerCommentRequest(BaseModel):
    """Запрос для генерации ответа на комментарий."""
    comment: str = Field(..., description="Комментарий для ответа")
    style: Optional[str] = Field("дружелюбном", description="Стиль ответа")
