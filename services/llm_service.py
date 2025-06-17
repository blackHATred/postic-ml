"""Сервис для работы с LLM."""
import asyncio
from ollama import AsyncClient
from config.settings import (
    OLLAMA_HOST, OLLAMA_TIMEOUT, OLLAMA_MODEL, 
    OLLAMA_EMBEDDING_MODEL, NUM_CTX, OLLAMA_KEEP_ALIVE
)


async def ollama_chat_completion(messages, temperature=1.0, seed=None):
    """Выполняет запрос к Ollama для генерации ответа."""
    options = {"temperature": temperature, "num_ctx": NUM_CTX}
    if seed is not None:
        options["seed"] = seed
    
    response = await AsyncClient(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT).chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options=options,
        keep_alive=OLLAMA_KEEP_ALIVE,  # Добавляем keep_alive для удержания модели в памяти
    )
    return response['message']['content']


async def ollama_chat_completion_stream(messages, temperature=1.0, seed=None):
    """Выполняет потоковый запрос к Ollama для генерации ответа."""
    options = {"temperature": temperature, "num_ctx": NUM_CTX}
    if seed is not None:
        options["seed"] = seed
    
    client = AsyncClient(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
    
    async for chunk in await client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options=options,
        keep_alive=OLLAMA_KEEP_ALIVE,
        stream=True,  # Включаем потоковый режим
    ):
        if 'message' in chunk and 'content' in chunk['message']:
            yield chunk['message']['content']


async def preload_models():
    """Предзагружает модели в память Ollama для ускорения последующих запросов."""
    client = AsyncClient(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
    
    try:
        print("🚀 Предзагрузка моделей Ollama...")
        
        # Предзагружаем основную модель
        print(f"⏳ Загружаем модель: {OLLAMA_MODEL}")
        await client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "test"}],
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        print(f"✅ Модель {OLLAMA_MODEL} загружена")
        
        # Предзагружаем модель эмбеддингов
        print(f"⏳ Загружаем модель эмбеддингов: {OLLAMA_EMBEDDING_MODEL}")
        await client.embeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            prompt="test",
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        print(f"✅ Модель эмбеддингов {OLLAMA_EMBEDDING_MODEL} загружена")
        
        print("🎉 Все модели успешно предзагружены и закреплены в памяти!")
        
    except Exception as e:
        print(f"⚠️ Ошибка при предзагрузке моделей: {e}")
        print("Приложение продолжит работу, но первые запросы могут быть медленными")


async def check_model_status():
    """Проверяет статус загруженных моделей."""
    client = AsyncClient(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
    
    try:
        # Получаем список запущенных моделей
        response = await client.list()
        models = response.get('models', [])
        
        loaded_models = []
        for model in models:
            if model.get('name') in [OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL]:
                loaded_models.append(model.get('name'))
        
        return {
            "chat_model_loaded": OLLAMA_MODEL in loaded_models,
            "embedding_model_loaded": OLLAMA_EMBEDDING_MODEL in loaded_models,
            "total_loaded": len(loaded_models),
            "models": loaded_models
        }
    except Exception as e:
        return {"error": str(e)}
