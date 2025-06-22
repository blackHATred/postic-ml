"""API эндпоинты для работы с публикациями - новая умная версия."""
import json
import math
import hashlib
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from config.settings import (
    SEARCH_K_COEF, TEMP, ENABLE_TIME_CONTEXT, 
    MAX_SEARCH_QUERIES, NUM_CTX
)
from services.search_service import index, get_relevant_documents
from services.llm_service import ollama_chat_completion, ollama_chat_completion_stream
from services.intelligent_query_service import IntelligentQueryService
from utils.text_processing import delete_all_links
from utils.timing import timer
from utils.context_utils import get_current_context, format_context_for_llm
from utils.strip_markdown import strip_markdown_and_links
from models.requests import PublicationRequest
from models.responses import PublicationResponse


def is_bad_image_url(url: str) -> bool:
    """Проверяет, является ли URL плохим изображением."""
    if not url or not isinstance(url, str):
        return True
    
    url_lower = url.lower()
    
    # Плохие паттерны в URL
    bad_patterns = [
        "logo", "icon", "banner", "ad", "promo", "sprite", "favicon",
        "default", "placeholder", "blank", "share", "social", "og_image",
        "avatar", "profile", "thumb_", "button", "header", "footer"
    ]
    
    # Проверяем плохие паттерны
    if any(pat in url_lower for pat in bad_patterns):
        return True
    
    # Проверяем, что это действительно изображение
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg']
    has_image_extension = any(ext in url_lower for ext in image_extensions)
    
    # Если нет расширения изображения и нет признаков изображения в URL - плохой URL
    image_indicators = ['images/', 'img/', 'media/', 'photo', 'picture', 'format=']
    has_image_indicator = any(indicator in url_lower for indicator in image_indicators)
    
    if not has_image_extension and not has_image_indicator:
        return True
    
    return False


def is_direct_image_url(url: str) -> bool:
    """Проверяет, является ли URL прямой ссылкой на изображение."""
    if not url or not isinstance(url, str):
        return False
    
    url_lower = url.lower()
    
    # Прямые расширения изображений
    direct_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
    
    for ext in direct_extensions:
        if url_lower.endswith(ext) or f"{ext}?" in url_lower:
            return True
    
    return False


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

            # Выполняем поиск по каждому запросу (только текст, без картинок)
            all_search_content = []
            for i, search_query in enumerate(search_queries, 1):
                print(f"🔍 Поиск {i}/{len(search_queries)}: {search_query}")
                with timer.measure(f"Индексация и поиск {i}"):
                    hash_name, chunk_count = await index(client, searcher, search_query)
                    chunk_count = int(chunk_count or 0)
                    if isinstance(hash_name, dict) and hash_name.get("llm_fallback"):
                        print(f"[index] Fallback на LLM для '{search_query}'")
                        continue
                    if not hash_name:
                        continue
                    search_k = math.ceil((chunk_count or 5) * SEARCH_K_COEF)
                    result = await get_relevant_documents(client, hash_name, search_query, search_k)
                    if not result or len(result) != 3:
                        raise HTTPException(status_code=400, detail=f"get_relevant_documents failed for '{search_query}'")
                    _, texts, _ = result  # images больше не используются
                    client.delete_collection(hash_name)
                    if texts and len(texts) > 0:
                        annotated_content = f"[Поиск: '{search_query}']\n" + "\n\n".join(texts[:2])
                        all_search_content.append(annotated_content)

            # Генерируем отдельные image queries через LLM
            with timer.measure("Генерация image-запросов через LLM"):
                image_queries = await query_service.generate_image_queries(query, 3)
                print(f"🖼️ LLM сгенерировал image-запросы: {image_queries}")            # Выполняем отдельный поиск по картинкам через SearX Images
            from config.settings import MAX_IMAGES_PER_QUERY, MAX_TOTAL_IMAGES
            
            all_images = []
            for img_query in image_queries:
                print(f"🖼️ Поиск картинки: {img_query}")
                img_results = await searcher.searx.search_images(img_query, num_results=MAX_IMAGES_PER_QUERY)
                for res in img_results:
                    img_url = res.get("img_src") or res.get("url")
                    if img_url and is_direct_image_url(img_url) and not is_bad_image_url(img_url):
                        all_images.append(img_url)
                        if len(all_images) >= MAX_TOTAL_IMAGES + 5:  # Небольшой запас для фильтрации
                            break
                if len(all_images) >= MAX_TOTAL_IMAGES + 5:
                    break
            
            # Удаляем дубликаты и ограничиваем количество
            all_images = list(dict.fromkeys(all_images))[:MAX_TOTAL_IMAGES]

            if not all_search_content:
                print("[index] Нет релевантной информации, но LLM всё равно сгенерирует ответ!")
                combined_content = ""
            else:
                # Объединяем весь найденный контент
                combined_content = delete_all_links("\n\n".join(all_search_content))
            print(f"📊 Обработано материалов: {len(all_search_content)} блоков")

            with timer.measure("Генерация финального ответа через LLM"):
                # Создаем итоговый промпт для LLM
                context_info = ""
                if time_context and ENABLE_TIME_CONTEXT:
                    context_info = format_context_for_llm(time_context) + "\n\n"

                # Дополнительные инструкции для LLM после предоставления информации
                extra_instructions = ("ВАЖНО:\n"
                    "- Не добавляй никаких ссылок, markdown, списков, заголовков, дополнительных ресурсов, фотографий, рекомендаций сайтов.\n"
                    "- Пиши только текст поста, как будто ты человек, без формальностей и шаблонов.\n"
                    "- НИКОГДА не начинай ответ с фраз типа 'Окей, вот пост', 'Вот пост', 'Держи пост', 'Готово' или любых других вступлений.\n"
                    "- Сразу начинай с содержания поста.\n"
                    "- При запросе подробной информации, рецептов, инструкций — предоставляй РАЗВЕРНУТЫЙ и ДЕТАЛЬНЫЙ ответ.\n"
                    "- Если ты добавишь ссылки, markdown или дополнительные ресурсы — это будет ошибкой.\n"
                    "- ЕЩЁ РАЗ: никаких ссылок, markdown, списков, заголовков, дополнительных ресурсов!\n"
                )

                system_content = (
                    f"{context_info}Ты — профессиональный копирайтер и контент-мейкер.\n\n"
                    "Твоя задача — создать качественный пост на основе найденной информации, точно выполняя запрос пользователя.\n\n"
                    "ПРАВИЛА:\n"
                    "- Всегда ставь пожелания пользователя на первое место. Если пользователь просит что-то особенное (эмодзи, стиль, длина, формат) — обязательно выполни это.\n"
                    "- Если пользователь просит короткий пост — делай его максимально кратким, не более 2-3 предложений.\n"
                    "- Если пользователь просит эмодзи, обязательно используй их. Если считаешь, что эмодзи уместны — добавь их.\n"
                    "- Не добавляй заголовки, списки, ссылки, markdown, дополнительные ресурсы, фотографии, рекомендации сайтов.\n"
                    "- Не добавляй вступления, пояснения, не повторяй запрос пользователя.\n"
                    "- Используй контекст по дате и времени ТОЛЬКО для поиска и выбора наиболее актуальной информации, но НЕ для явного упоминания даты/времени в тексте поста, если это не требуется пользователем.\n"
                    "- Если запрашивают эмодзи — используй их умеренно.\n"
                    "- Пост должен быть БЕЗ markdown-разметки и БЕЗ ссылок.\n"
                    "- Если в исходном тексте есть markdown-разметка, обязательно убери её в ответе.\n"
                    "- Используй информацию из разных источников для полноты.\n"
                    "- Делай пост информативным и интересным для чтения.\n"
                    "- При запросе рецептов, инструкций, руководств — предоставляй ПОЛНУЮ информацию со всеми деталями.\n"
                    "- НИКОГДА не начинай ответ с фраз типа 'Окей, вот пост', 'Вот пост', 'Держи пост', 'Готово', или любых других вступлений.\n"
                    "- Сразу начинай с содержания поста, без предисловий и объяснений.\n"
                    "- Если пользователь просит подробную информацию, рецепт, инструкцию или пошаговое руководство — создавай РАЗВЕРНУТЫЙ и ДЕТАЛЬНЫЙ пост с полной информацией.\n"
                    f"{extra_instructions}"                )

                # Создаем структуру сообщений: система → контекст → система → задача
                context_message = f"НАЙДЕННАЯ ИНФОРМАЦИЯ:\n{combined_content}" if combined_content else "Дополнительная информация не найдена, используй общие знания."
                
                task_message = (
                    f"ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {query}\n\n"
                    f"{extra_instructions}"
                    "Создай пост, который точно отвечает на запрос пользователя, используя предоставленную выше информацию."
                )

                # Убираем ограничение длины - позволяем Ollama самой управлять контекстом
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": context_message},
                    {"role": "system", "content": system_content},  # Дублируем для усиления инструкций
                    {"role": "user", "content": task_message},
                ]

                # Генерируем финальный ответ
                response = await ollama_chat_completion(messages, temperature=TEMP, seed=420)
                # Постпроцессинг: удаляем markdown и ссылки
                response = strip_markdown_and_links(response)
            print(f"📝 Результат сгенерирован ({len(response)} символов)")
            with timer.measure("Сохранение в кеш"):
                json_content = {
                    "text": response, 
                    "images": all_images,
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
        
        # Инициализируем умный сервис
        query_service = IntelligentQueryService()
        
        # Генерируем умные поисковые запросы через LLM
        search_queries = await query_service.generate_search_queries(query, MAX_SEARCH_QUERIES)
        yield f"data: {json.dumps({'type': 'queries', 'message': f'LLM сгенерировал запросы: {search_queries}', 'queries': search_queries})}\n\n"

        # Выполняем поиск по каждому запросу (только текст, без картинок)
        all_search_content = []
        for i, search_query in enumerate(search_queries, 1):
            yield f"data: {json.dumps({'type': 'search', 'message': f'Поиск {i}/{len(search_queries)}: {search_query}'})}\n\n"
            
            # Выполняем индексацию и поиск
            hash_name, chunk_count = await index(client, searcher, search_query)
            chunk_count = int(chunk_count or 0)
            # Fallback на LLM, если не найдено
            if isinstance(hash_name, dict) and hash_name.get("llm_fallback"):
                yield f"data: {json.dumps({'type': 'warning', 'message': f'Нет релевантных результатов, fallback на LLM для: {search_query}'})}\n\n"
                continue
            if not hash_name:
                yield f"data: {json.dumps({'type': 'warning', 'message': f'Не удалось проиндексировать: {search_query}'})}\n\n"
                continue

            search_k = math.ceil((chunk_count or 5) * SEARCH_K_COEF)
            result = await get_relevant_documents(client, hash_name, search_query, search_k)
            if not result or len(result) != 3:
                yield f"data: {json.dumps({'type': 'error', 'message': f'get_relevant_documents вернул None или некорректный результат для: {search_query}'})}\n\n"
                continue
            _, texts, _ = result  # images больше не используются

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
            yield f"data: {json.dumps({'type': 'warning', 'message': 'Нет релевантной информации, но LLM всё равно сгенерирует ответ!'})}\n\n"
            combined_content = ""
        else:
            yield f"data: {json.dumps({'type': 'processing', 'message': f'Обработано материалов: {len(all_search_content)} блоков'})}\n\n"
            # Объединяем весь найденный контент
            combined_content = delete_all_links("\n\n".join(all_search_content))

        # --- Новый этап: отдельная генерация image queries и поиск картинок ---
        with timer.measure("Генерация image-запросов через LLM"):
            image_queries = await query_service.generate_image_queries(query, 3)
            yield f"data: {json.dumps({'type': 'image_queries', 'message': f'LLM сгенерировал image-запросы: {image_queries}', 'queries': image_queries})}\n\n"

        from config.settings import MAX_IMAGES_PER_QUERY, MAX_TOTAL_IMAGES
        
        all_images = []
        for img_query in image_queries:
            yield f"data: {json.dumps({'type': 'image_search', 'message': f'Поиск картинки: {img_query}'})}\n\n"
            img_results = await searcher.searx.search_images(img_query, num_results=MAX_IMAGES_PER_QUERY)
            for res in img_results:
                img_url = res.get("img_src") or res.get("url")
                if img_url and is_direct_image_url(img_url) and not is_bad_image_url(img_url):
                    all_images.append(img_url)
                    if len(all_images) >= MAX_TOTAL_IMAGES + 5:  # Небольшой запас для фильтрации
                        break
            if len(all_images) >= MAX_TOTAL_IMAGES + 5:
                break
        
        # Удаляем дубликаты и ограничиваем количество
        all_images = list(dict.fromkeys(all_images))[:MAX_TOTAL_IMAGES]

        # Формируем промпт для финальной генерации
        context_info = ""
        if time_context and ENABLE_TIME_CONTEXT:
            context_info = format_context_for_llm(time_context) + "\n\n"        # Дополнительные инструкции для LLM после предоставления информации
        extra_instructions = ("ВАЖНО:\n"
            "- Не добавляй никаких ссылок, markdown, списков, заголовков, дополнительных ресурсов, фотографий, рекомендаций сайтов.\n"
            "- Пиши только текст поста, как будто ты человек, без формальностей и шаблонов.\n"
            "- НИКОГДА не начинай ответ с фраз типа 'Окей, вот пост', 'Вот пост', 'Держи пост', 'Готово' или любых других вступлений.\n"
            "- Сразу начинай с содержания поста.\n"
            "- При запросе подробной информации, рецептов, инструкций — предоставляй РАЗВЕРНУТЫЙ и ДЕТАЛЬНЫЙ ответ.\n"
            "- Если ты добавишь ссылки, markdown или дополнительные ресурсы — это будет ошибкой.\n"
            "- ЕЩЁ РАЗ: никаких ссылок, markdown, списков, заголовков, дополнительных ресурсов!\n"
        )

        system_content = (
            f"{context_info}Ты — профессиональный копирайтер и контент-мейкер.\n\n"
            "Твоя задача — создать качественный пост на основе найденной информации, точно выполняя запрос пользователя.\n\n"
            "ПРАВИЛА:\n"
            "- Всегда ставь пожелания пользователя на первое место. Если пользователь просит что-то особенное (эмодзи, стиль, длина, формат) — обязательно выполни это.\n"
            "- Если пользователь просит короткий пост — делай его максимально кратким, не более 2-3 предложений.\n"
            "- Если пользователь просит эмодзи, обязательно используй их. Если считаешь, что эмодзи уместны — добавь их.\n"
            "- Не добавляй заголовки, списки, ссылки, markdown, дополнительные ресурсы, фотографии, рекомендации сайтов.\n"
            "- Не добавляй вступления, пояснения, не повторяй запрос пользователя.\n"
            "- Используй контекст по дате и времени ТОЛЬКО для поиска и выбора наиболее актуальной информации, но НЕ для явного упоминания даты/времени в тексте поста, если это не требуется пользователем.\n"
            "- Если запрашивают эмодзи — используй их умеренно.\n"
            "- Пост должен быть БЕЗ markdown-разметки и БЕЗ ссылок.\n"
            "- Если в исходном тексте есть markdown-разметка, обязательно убери её в ответе.\n"
            "- Используй информацию из разных источников для полноты.\n"
            "- Делай пост информативным и интересным для чтения.\n"
            "- При запросе рецептов, инструкций, руководств — предоставляй ПОЛНУЮ информацию со всеми деталями.\n"
            "- НИКОГДА не начинай ответ с фраз типа 'Окей, вот пост', 'Вот пост', 'Держи пост', 'Готово', или любых других вступлений.\n"
            "- Сразу начинай с содержания поста, без предисловий и объяснений.\n"
            "- Если пользователь просит подробную информацию, рецепт, инструкцию или пошаговое руководство — создавай РАЗВЕРНУТЫЙ и ДЕТАЛЬНЫЙ пост с полной информацией.\n"
            f"{extra_instructions}"        )

        # Создаем структуру сообщений: система → контекст → система → задача
        context_message = f"НАЙДЕННАЯ ИНФОРМАЦИЯ:\n{combined_content}" if combined_content else "Дополнительная информация не найдена, используй общие знания."
        
        task_message = (
            f"ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {query}\n\n"
            f"{extra_instructions}"
            "Создай пост, который точно отвечает на запрос пользователя, используя предоставленную выше информацию."
        )

        # Убираем ограничение длины - позволяем Ollama самой управлять контекстом
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": context_message},
            {"role": "system", "content": system_content},  # Дублируем для усиления инструкций
            {"role": "user", "content": task_message},
        ]

        yield f"data: {json.dumps({'type': 'generation', 'message': 'Генерируем финальный ответ через LLM...', 'images': None})}\n\n"

        # Потоковая генерация ответа
        full_response = ""
        async for chunk in ollama_chat_completion_stream(messages, temperature=TEMP, seed=420, chunk_size=64):
            if chunk:
                full_response += chunk
                # Отправляем каждый накопленный кусок текста
                yield f"data: {json.dumps({'type': 'content', 'text': chunk, 'final': False})}\n\n"

        # Постпроцессинг: удаляем markdown и ссылки
        full_response = strip_markdown_and_links(full_response)

        # Используем уже собранный all_images (до генерации)
        # Финальное сообщение
        yield f"data: {json.dumps({'type': 'content', 'text': '', 'final': True, 'full_text': full_response, 'images': all_images})}\n\n"

        # Сохраняем в кеш
        json_content = {
            "text": full_response, 
            "images": all_images,
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
