"""API эндпоинты для работы с публикациями."""
import json
import math
import hashlib
from fastapi import HTTPException, Request

from config.settings import SEARCH_K_COEF, TEMP
from services.search_service import index, get_relevant_documents
from services.llm_service import ollama_chat_completion
from utils.text_processing import delete_all_links


async def get_publication(request: Request, client, redis_client, searcher):
    """Генерирует пост на основе поискового запроса."""
    try:
        payload = await request.json()
        query = payload["query"]
        md5_hash = hashlib.new('md5')
        md5_hash.update((query).encode())
        key = "publication-" + md5_hash.hexdigest()
        cached = redis_client.get(key)
        
        if cached is None:
            hash_name, chunk_count = await index(client, searcher, query)
            if hash_name is None:
                raise Exception("Индексация - FAIL")
            
            _, texts, images = await get_relevant_documents(
                client, hash_name, query, math.ceil(chunk_count * SEARCH_K_COEF)
            )
            
            if not client.delete_collection(hash_name):
                raise Exception("Удаление временных индексов - FAIL")
            
            images = list(images[0].values())
            if len(images) > 0:
                imgs_ret = [images[0]]
                for i in range(len(images)):
                    if images[i] != imgs_ret[-1]:
                        imgs_ret.append(images[i])
            else:
                imgs_ret = None
            
            text = delete_all_links(delete_all_links(texts[0]))
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"{text}\nСгенерируй пост на основе текста выше, будто я сам это пишу. "
                        f"Никакой разметки. Никаких ссылок. Только живой, личный текст от первого лица."
                    )
                },
                {
                    "role": "system",
                    "content": (
                        "Ты — копирайтер, создающий посты для социальных сетей. "
                        "Ты пишешь короткий, живой и личный пост **от первого лица**. "
                        "Пост должен быть написан **без markdown-разметки**, **без ссылок**, **без изображений** и **без отстранённых комментариев**. "
                        "Не используй фразы вроде 'в этом тексте говорится' или 'в данном документе'. "
                        "Цель — вовлечь читателя, а не анализировать текст."
                    )
                },
            ]
            response = await ollama_chat_completion(messages, temperature=TEMP, seed=420)
            json_content = {"text": response, "images": imgs_ret}
            redis_client.set(key, json.dumps(json_content))
        else:
            json_content = json.loads(cached)
        
        return json_content
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
