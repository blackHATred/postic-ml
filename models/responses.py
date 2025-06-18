"""Модели ответов для API эндпоинтов."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PublicationResponse(BaseModel):
    """Ответ для генерации публикации."""
    text: str = Field(..., description="Сгенерированный текст публикации")
    images: Optional[List[str]] = Field(None, description="Список изображений для публикации")


class SentimentResponse(BaseModel):
    """Ответ для анализа тональности."""
    label: str = Field(..., description="Метка тональности (POSITIVE, NEGATIVE, NEUTRAL)")
    score: float = Field(..., description="Уверенность модели в предсказании")


class TicketResponse(BaseModel):
    """Ответ для определения необходимости поддержки."""
    support_needed: bool = Field(..., description="Нужна ли поддержка")


class SummaryResponse(BaseModel):
    """Ответ для суммаризации комментариев."""
    response: str = Field(..., description="Краткое содержание комментариев")


class FixTextResponse(BaseModel):
    """Ответ для исправления текста."""
    response: str = Field(..., description="Исправленный текст")


class AnswerCommentResponse(BaseModel):
    """Ответ для генерации ответа на комментарий."""
    no_answer: bool = Field(..., description="Не нужно отвечать (спам/оскорбления)")
    support_needed: bool = Field(..., description="Нужно направить в поддержку")
    answer_0: Optional[str] = Field(None, description="Первый вариант ответа")
    answer_1: Optional[str] = Field(None, description="Второй вариант ответа")
    answer_2: Optional[str] = Field(None, description="Третий вариант ответа")


class ErrorResponse(BaseModel):
    """Стандартный ответ об ошибке."""
    detail: str = Field(..., description="Описание ошибки")
