"""Сервис для умной генерации поисковых запросов через LLM."""
import asyncio
import json
import re
from typing import List, Dict, Any
from services.llm_service import ollama_chat_completion
from utils.context_utils import get_current_context


class IntelligentQueryService:
    """Умный сервис для генерации поисковых запросов через LLM с учетом контекста."""
    
    def __init__(self):
        self.max_retries = 2
        
    async def generate_search_queries(self, user_query: str, max_queries: int = 3) -> List[str]:
        """
        Генерирует оптимальные поисковые запросы через LLM с учетом контекста времени.
        """
        context = get_current_context()
        
        # Формируем контекстную информацию
        context_info = ""
        if context:
            context_info = f"""
ТЕКУЩИЙ КОНТЕКСТ:
- Дата: {context['current_datetime']}
- День недели: {context['day_of_week']}
- Сезон: {context['season']}
- Месяц: {context['current_month_name']}
- Год: {context['current_year']}
"""

        system_prompt = f"""Ты — эксперт по поисковым запросам. Твоя задача — создать {max_queries} оптимальных поисковых запроса для поиска актуальной информации в интернете.

{context_info}

ПРАВИЛА СОЗДАНИЯ ПОИСКОВЫХ ЗАПРОСОВ:
1. Анализируй намерения пользователя и извлекай ключевые темы
2. Создавай запросы разной специфичности (общие и конкретные)
3. Учитывай текущую дату для актуальности, если это уместно
4. Используй синонимы и альтернативные формулировки
5. Для новостных тем добавляй временные маркеры (2025, сегодня, последние)
6. Делай запросы краткими но информативными (3-7 слов)
7. НЕ дублируй запросы, каждый должен быть уникальным

ФОРМАТ ОТВЕТА:
Отвечай ТОЛЬКО списком JSON с полями "queries". Никакого дополнительного текста!

Пример:
{{"queries": ["запрос 1", "запрос 2", "запрос 3"]}}"""

        user_prompt = f'Создай поисковые запросы для: "{user_query}"'

        for attempt in range(self.max_retries + 1):
            try:
                response = await ollama_chat_completion([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ], temperature=0.7)
                
                # Парсим JSON ответ
                queries = self._parse_llm_response(response)
                
                if queries and len(queries) > 0:
                    # Валидируем и очищаем запросы
                    valid_queries = self._validate_queries(queries, user_query)
                    
                    if valid_queries:
                        print(f"✅ Сгенерировано {len(valid_queries)} поисковых запросов: {valid_queries}")
                        return valid_queries[:max_queries]
                
                print(f"⚠️ Попытка {attempt + 1}: LLM вернул некорректные запросы")
                
            except Exception as e:
                print(f"⚠️ Ошибка в попытке {attempt + 1}: {e}")
                
        # Fallback: создаем простые запросы на основе ключевых слов
        print("🔄 Используем fallback метод генерации запросов")
        return self._generate_fallback_queries(user_query, max_queries)
    
    async def generate_image_queries(self, user_query: str, max_queries: int = 3) -> List[str]:
        """
        Генерирует поисковые запросы для поиска релевантных изображений через LLM.
        """
        context = get_current_context()
        context_info = ""
        if context:
            context_info = f"""
ТЕКУЩИЙ КОНТЕКСТ:
- Дата: {context['current_datetime']}
- День недели: {context['day_of_week']}
- Сезон: {context['season']}
- Месяц: {context['current_month_name']}
- Год: {context['current_year']}
"""

        system_prompt = f"""Ты — эксперт по поисковым запросам для поиска изображений. Твоя задача — создать {max_queries} оптимальных поисковых запроса для поиска релевантных картинок в интернете по теме пользователя.

{context_info}

ПРАВИЛА СОЗДАНИЯ ЗАПРОСОВ:
1. Анализируй намерения пользователя и тему
2. Формируй запросы так, чтобы найти подходящие иллюстрации, фотографии, инфографику, эмодзи, мемы, если это уместно
3. Добавляй слова: фото, картинка, иллюстрация, инфографика, эмодзи, мем, если это подходит
4. Для новостных тем добавляй временные маркеры (2025, сегодня, последние)
5. Делай запросы краткими и информативными (3-7 слов)
6. НЕ дублируй запросы, каждый должен быть уникальным

ФОРМАТ ОТВЕТА:
Отвечай ТОЛЬКО списком JSON с полями "queries". Никакого дополнительного текста!

Пример:
{{"queries": ["фото доллар США 2025", "инфографика экономика", "эмодзи деньги"]}}"""

        user_prompt = f'Создай поисковые запросы для поиска картинок по теме: "{user_query}"'

        for attempt in range(self.max_retries + 1):
            try:
                response = await ollama_chat_completion([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ], temperature=0.7)
                queries = self._parse_llm_response(response)
                if queries and len(queries) > 0:
                    valid_queries = self._validate_queries(queries, user_query)
                    if valid_queries:
                        print(f"✅ Сгенерировано {len(valid_queries)} image-запросов: {valid_queries}")
                        return valid_queries[:max_queries]
                print(f"⚠️ Попытка {attempt + 1}: LLM вернул некорректные image-запросы")
            except Exception as e:
                print(f"⚠️ Ошибка в попытке {attempt + 1} (image queries): {e}")
        print("🔄 Используем fallback метод генерации image-запросов")
        return self._generate_fallback_queries(user_query, max_queries)
    
    def _parse_llm_response(self, response: str) -> List[str]:
        """Парсит ответ LLM и извлекает поисковые запросы."""
        try:
            # Ищем JSON в ответе
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('queries', [])
            
            # Если JSON не найден, пробуем парсить как список
            lines = response.strip().split('\n')
            queries = []
            for line in lines:
                line = line.strip()
                # Убираем маркеры списков
                line = re.sub(r'^[-•*\d+\.)\s]+', '', line)
                if line and len(line) > 3:
                    queries.append(line)
            return queries
            
        except Exception as e:
            print(f"Ошибка парсинга ответа LLM: {e}")
            return []
    
    def _validate_queries(self, queries: List[str], original_query: str) -> List[str]:
        """Валидирует и очищает сгенерированные запросы."""
        valid_queries = []
        seen = set()
        
        for query in queries:
            # Очищаем запрос
            cleaned = self._clean_query(query)
            
            # Проверяем валидность
            if (len(cleaned) >= 3 and len(cleaned) <= 100 and 
                cleaned.lower() not in seen and
                not cleaned.lower() in original_query.lower()):
                
                valid_queries.append(cleaned)
                seen.add(cleaned.lower())
                
        return valid_queries
    
    def _clean_query(self, query: str) -> str:
        """Очищает поисковый запрос от лишних символов."""
        # Убираем кавычки и специальные символы
        query = re.sub(r'["\'\[\]{}()]', '', query)
        # Убираем множественные пробелы
        query = re.sub(r'\s+', ' ', query)
        return query.strip()
    
    def _generate_fallback_queries(self, user_query: str, max_queries: int) -> List[str]:
        """Создает простые поисковые запросы как fallback."""
        context = get_current_context()
        current_year = context['current_year'] if context else "2025"
        
        # Извлекаем ключевые слова
        words = re.findall(r'\b\w+\b', user_query.lower())
        important_words = [w for w in words if len(w) > 3 and w not in {
            'напиши', 'создай', 'расскажи', 'пост', 'статью', 'текст', 'публикацию'
        }]
        
        queries = []
        
        if important_words:
            # Основной запрос с ключевыми словами
            main_query = ' '.join(important_words[:4])
            queries.append(f"{main_query} {current_year}")
            
            # Более специфичный запрос
            if len(important_words) >= 2:
                specific_query = ' '.join(important_words[:2])
                queries.append(f"{specific_query} новости {current_year}")
            
            # Общий запрос
            if len(important_words) >= 1:
                general_query = important_words[0]
                queries.append(f"{general_query} тренды {current_year}")
        
        # Если ничего не получилось, используем исходный запрос
        if not queries:
            queries = [f"{user_query} {current_year}"]
            
        return queries[:max_queries]
