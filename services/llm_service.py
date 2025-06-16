"""Сервис для работы с LLM."""
import asyncio
from ollama import AsyncClient
from config.settings import OLLAMA_HOST, OLLAMA_TIMEOUT, OLLAMA_MODEL, NUM_CTX


async def ollama_chat_completion(messages, temperature=1.0, seed=None):
    """Выполняет запрос к Ollama для генерации ответа."""
    options = {"temperature": temperature, "num_ctx": NUM_CTX}
    if seed is not None:
        options["seed"] = seed
    
    response = await AsyncClient(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT).chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options=options,
    )
    return response['message']['content']
