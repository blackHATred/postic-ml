"""API эндпоинты для работы с публикациями - новая умная версия."""
import json
import math
import hashlib
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from config.settings import (
    SEARCH_K_COEF, TEMP, ENABLE_TIME_CONTEXT, 
    MAX_SEARCH_QUERIES
)
from services.search_service import index, get_relevant_documents
from services.llm_service import ollama_chat_completion, ollama_chat_completion_stream
from services.intelligent_query_service import IntelligentQueryService
from utils.text_processing import delete_all_links
from utils.timing import timer
from utils.context_utils import get_current_context, format_context_for_llm
from models.requests import PublicationRequest
from models.responses import PublicationResponse


async def get_publication(request: PublicationRequest, client, redis_client, searcher) -> PublicationResponse:
    """Генерирует пост используя умный поиск через LLM."""
    try:
        query = request.query
        timer.reset()
        
        print(f"\n{'='*80}")
        print(f"📝 НОВЫЙ ЗАПРОС НА ПУБЛИКАЦИЮ: {query}")
        print(f"{'='*80}")
        
        # Получаем контекст времени
        time_context = get_current_context() if ENABLE_TIME_CONTEXT else None
        if time_context:
            print(f"⏰ Контекст времени: {time_context['current_datetime']} ({time_context['day_of_week']})")
        
        # Проверяем кеш
        with timer.measure("Кеширование"):
            cache_key_data = f"{query}_{time_context['current_date'] if time_context else ''}"
            md5_hash = hashlib.new('md5')
            md5_hash.update(cache_key_data.encode())
            key = "publication-" + md5_hash.hexdigest()
            cached = redis_client.get(key)

        if cached is None:
            print(f"🔄 Генерируем новый контент...")

            # Инициализируем новый умный сервис
            query_service = IntelligentQueryService()
            
            with timer.measure("Генерация поисковых запросов через LLM"):
                # Генерируем умные поисковые запросы через LLM
                search_queries = await query_service.generate_search_queries(query, MAX_SEARCH_QUERIES)
                print(f"🎯 LLM сгенерировал поисковые запросы: {search_queries}")

            # Выполняем поиск по каждому запросу
            all_search_content = []
            for i, search_query in enumerate(search_queries, 1):
                print(f"🔍 Поиск {i}/{len(search_queries)}: {search_query}")

                with timer.measure(f"Индексация и поиск {i}"):
                    hash_name, chunk_count = await index(client, searcher, search_query)
                    if hash_name is None:
                        print(f"❌ Индексация запроса '{search_query}' неудачна")
                        continue

                    search_k = math.ceil((chunk_count or 5) * SEARCH_K_COEF)
                    _, texts, images = await get_relevant_documents(client, hash_name, search_query, search_k)

                    # Очистка временной коллекции
                    if not client.delete_collection(hash_name):
                        print(f"⚠️ Не удалось удалить временную коллекцию {hash_name}")

                    # Сохраняем найденные материалы
                    if texts and len(texts) > 0:
                        # Аннотируем контент источником
                        annotated_content = f"[Поиск: '{search_query}']\n" + "\n\n".join(texts[:2])
                        all_search_content.append(annotated_content)
                        print(f"✅ Найдено для '{search_query}': {len(texts)} текстов")
                    else:
                        print(f"❌ Нет результатов для '{search_query}'")

            if not all_search_content:
                raise Exception("Не удалось найти релевантную информацию по запросу")

            # Объединяем весь найденный контент
            combined_content = delete_all_links("\n\n".join(all_search_content))
            print(f"📊 Обработано материалов: {len(all_search_content)} блоков")

            with timer.measure("Генерация финального ответа через LLM"):
                # Создаем итоговый промпт для LLM
                context_info = ""
                if time_context and ENABLE_TIME_CONTEXT:
                    context_info = format_context_for_llm(time_context) + "\n\n"

                system_content = f"""{context_info}Ты — профессиональный копирайтер и контент-мейкер.

Твоя задача — создать качественный пост на основе найденной информации, точно выполняя запрос пользователя.

ПРАВИЛА:
- Анализируй найденную информацию и создавай на её основе оригинальный контент
- Адаптируй стиль под запрос (новости, мнение, обзор, и т.д.)
- Учитывай текущую дату и время при формулировке
- Если запрашивают эмодзи — используй их умеренно
- Пост должен быть БЕЗ markdown-разметки и БЕЗ ссылок
- Используй информацию из разных источников для полноты
- Делай пост информативным и интересным для чтения"""

                user_content = f"""НАЙДЕННАЯ ИНФОРМАЦИЯ:
{combined_content}

ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {query}

Создай пост, который точно отвечает на запрос пользователя, используя найденную информацию."""

                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ]

                # Генерируем финальный ответ
                response = await ollama_chat_completion(messages, temperature=TEMP, seed=420)
                
            print(f"📝 Результат сгенерирован ({len(response)} символов)")
            
            with timer.measure("Сохранение в кеш"):
                json_content = {
                    "text": response, 
                    "images": None,  # Пока без изображений для упрощения
                    "search_queries_used": search_queries,
                    "sources_count": len(all_search_content),
                    "generated_at": time_context['current_datetime'] if time_context else None
                }
                cache_ttl = 1800 if time_context else 3600
                redis_client.setex(key, cache_ttl, json.dumps(json_content))
        else:
            with timer.measure("Загрузка из кеша"):
                json_content = json.loads(cached)
                print(f"✅ Использован кешированный результат (создан: {json_content.get('generated_at', 'неизвестно')})")

        # Выводим итоговую таблицу времени
        print("\n" + timer.get_summary_table())
        print(f"{'='*80}\n")
        
        return PublicationResponse(text=json_content["text"], images=json_content.get("images"))
        
    except Exception as e:
        print(f"❌ Ошибка в get_publication: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


async def get_publication_stream(request: PublicationRequest, client, redis_client, searcher) -> AsyncGenerator[str, None]:
    """Генерирует пост с потоковой отправкой ответа через SSE - новая версия."""
    try:
        query = request.query
        
        # Отправляем начальное сообщение
        yield f"data: {json.dumps({'type': 'status', 'message': 'Начинаем обработку запроса...', 'query': query})}\n\n"
        
        # Получаем контекст времени
        time_context = get_current_context() if ENABLE_TIME_CONTEXT else None
        if time_context:
            context_msg = f"Контекст времени: {time_context['current_datetime']}"
            yield f"data: {json.dumps({'type': 'context', 'message': context_msg, 'time_context': time_context})}\n\n"
        
        # Проверяем кеш
        cache_key_data = f"{query}_{time_context['current_date'] if time_context else ''}"
        md5_hash = hashlib.new('md5')
        md5_hash.update(cache_key_data.encode())
        key = "publication-" + md5_hash.hexdigest()
        cached = redis_client.get(key)

        if cached is not None:
            json_content = json.loads(cached)
            yield f"data: {json.dumps({'type': 'cache', 'message': 'Найден кешированный результат'})}\n\n"
            yield f"data: {json.dumps({'type': 'content', 'text': json_content['text'], 'images': json_content.get('images'), 'final': True})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'status', 'message': 'Генерируем умные поисковые запросы через LLM...'})}\n\n"
        
        # Инициализируем новый умный сервис
        query_service = IntelligentQueryService()
        
        # Генерируем умные поисковые запросы через LLM
        search_queries = await query_service.generate_search_queries(query, MAX_SEARCH_QUERIES)
        yield f"data: {json.dumps({'type': 'queries', 'message': f'LLM сгенерировал запросы: {search_queries}', 'queries': search_queries})}\n\n"

        # Выполняем поиск по каждому запросу
        all_search_content = []
        for i, search_query in enumerate(search_queries, 1):
            yield f"data: {json.dumps({'type': 'search', 'message': f'Поиск {i}/{len(search_queries)}: {search_query}'})}\n\n"
            
            # Выполняем индексацию и поиск
            hash_name, chunk_count = await index(client, searcher, search_query)
            if hash_name is None:
                yield f"data: {json.dumps({'type': 'warning', 'message': f'Не удалось проиндексировать: {search_query}'})}\n\n"
                continue

            search_k = math.ceil((chunk_count or 5) * SEARCH_K_COEF)
            _, texts, images = await get_relevant_documents(client, hash_name, search_query, search_k)

            # Очистка временной коллекции
            if not client.delete_collection(hash_name):
                yield f"data: {json.dumps({'type': 'warning', 'message': f'Не удалось удалить временную коллекцию {hash_name}'})}\n\n"

            # Сохраняем найденные материалы
            if texts and len(texts) > 0:
                annotated_content = f"[Поиск: '{search_query}']\n" + "\n\n".join(texts[:2])
                all_search_content.append(annotated_content)
                yield f"data: {json.dumps({'type': 'search_result', 'message': f'Найдено {len(texts)} результатов для: {search_query}'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'warning', 'message': f'Нет результатов для: {search_query}'})}\n\n"

        if not all_search_content:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Не удалось найти релевантную информацию по запросу'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'processing', 'message': f'Обработано материалов: {len(all_search_content)} блоков'})}\n\n"

        # Объединяем весь найденный контент
        combined_content = delete_all_links("\n\n".join(all_search_content))

        # Формируем промпт для финальной генерации
        context_info = ""
        if time_context and ENABLE_TIME_CONTEXT:
            context_info = format_context_for_llm(time_context) + "\n\n"

        system_content = f"""{context_info}Ты — профессиональный копирайтер и контент-мейкер.

Твоя задача — создать качественный пост на основе найденной информации, точно выполняя запрос пользователя.

ПРАВИЛА:
- Анализируй найденную информацию и создавай на её основе оригинальный контент
- Адаптируй стиль под запрос (новости, мнение, обзор, и т.д.)
- Учитывай текущую дату и время при формулировке
- Если запрашивают эмодзи — используй их умеренно
- Пост должен быть БЕЗ markdown-разметки и БЕЗ ссылок
- Используй информацию из разных источников для полноты
- Делай пост информативным и интересным для чтения"""

        user_content = f"""НАЙДЕННАЯ ИНФОРМАЦИЯ:
{combined_content}

ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {query}

Создай пост, который точно отвечает на запрос пользователя, используя найденную информацию."""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        yield f"data: {json.dumps({'type': 'generation', 'message': 'Генерируем финальный ответ через LLM...', 'images': None})}\n\n"

        # Потоковая генерация ответа
        full_response = ""
        async for chunk in ollama_chat_completion_stream(messages, temperature=TEMP, seed=420):
            if chunk:
                full_response += chunk
                # Отправляем каждый кусочек текста
                yield f"data: {json.dumps({'type': 'content', 'text': chunk, 'final': False})}\n\n"

        # Финальное сообщение
        yield f"data: {json.dumps({'type': 'content', 'text': '', 'final': True, 'full_text': full_response})}\n\n"

        # Сохраняем в кеш
        json_content = {
            "text": full_response, 
            "images": None,
            "search_queries_used": search_queries,
            "sources_count": len(all_search_content),
            "generated_at": time_context['current_datetime'] if time_context else None
        }
        cache_ttl = 1800 if time_context else 3600
        redis_client.setex(key, cache_ttl, json.dumps(json_content))

        yield f"data: {json.dumps({'type': 'complete', 'message': 'Генерация завершена'})}\n\n"
        
    except Exception as e:
        error_msg = f"Ошибка в get_publication_stream: {str(e)}"
        print(f"❌ {error_msg}")
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
