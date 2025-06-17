"""Сервис для работы с векторными вложениями."""
from typing import List
from ollama import AsyncClient
from config.settings import (
    OLLAMA_HOST, OLLAMA_TIMEOUT, OLLAMA_EMBEDDING_MODEL, OLLAMA_KEEP_ALIVE
)


async def get_vector(texts: List, images: List = None) -> List[float]:
    """Получает векторное представление для текста."""
    response = await AsyncClient(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT).embed(
        model=OLLAMA_EMBEDDING_MODEL,
        input=texts,
        keep_alive=OLLAMA_KEEP_ALIVE,  # Добавляем keep_alive для удержания модели в памяти
    )
    return response["embeddings"][0]
