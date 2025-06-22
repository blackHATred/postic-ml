"""Сервис для поиска через DuckDuckGo с простой обработкой рейт-лимитов."""
import json
import hashlib
import asyncio
import httpx
import redis
from bs4 import BeautifulSoup
from typing import Dict, List, Set
from markdownify import markdownify as md
from urllib.parse import urljoin, urlparse
from services.searx_search import SearxSearch
from config.settings import (
    SEARCH_TIME_RANGE, SEARCH_MULTIPLIER, SEARCH_REGION, 
    SEARCH_SAFESEARCH, TITLE_KEYWORD_BONUS, MIN_CONTENT_LENGTH
)
from utils.timing import timer


def first_word_with_number(text):
    """Извлекает первое слово с цифрой из текста."""
    words = text.split()
    for word in words:
        if any(char.isdigit() for char in word):
            return word
    return None


def filter_links_by_blacklist(links: List[str], blacklist: Set[str]) -> List[str]:
    """Фильтрует ссылки по черному списку доменов."""
    filtered_links = []
    for link in links:
        domain = urlparse(link).netloc.lower()
        blocked = False
        for blacklisted in blacklist:
            blacklisted = blacklisted.lower()
            if blacklisted.startswith("."):
                if domain.endswith(blacklisted):
                    blocked = True
                    break
            else:
                if domain == blacklisted:
                    blocked = True
                    break
        if not blocked:
            filtered_links.append(link)
    return filtered_links


class SearxWebSearch:
    """Сервис для поиска через Searx/SearxNG с ротацией публичных инстансов."""
    
    def __init__(self, redis_client: redis.Redis, ref_cnt: int, timeout: int, blacklist: Set[str], searx_instances=None):
        self.redis_client = redis_client
        self.ref_cnt = ref_cnt
        self.timeout = timeout
        self.blacklist = blacklist
        self.searx = SearxSearch(instances=searx_instances, timeout=2)
        self.client = httpx.AsyncClient(
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        print("SearxWebSearch initialized with ref_cnt:", ref_cnt, "timeout:", timeout)

    async def html_to_md(self, url: str) -> str:
        """Конвертирует HTML страницу в Markdown с простой обработкой ошибок."""
        try:
            response = await self.client.get(url=url, timeout=self.timeout)
            
            # Простая проверка на рейт-лимит
            if response.status_code == 429:
                print(f"🚫 Рейт-лимит при загрузке {url}, ждем 0.5 секунды")
                await asyncio.sleep(0.5)
                # Повторная попытка
                response = await self.client.get(url=url, timeout=self.timeout)
            
            if not response.is_success:
                print(f"⚠️ HTTP {response.status_code} для {url}")
                return ""
            
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Удаляем ненужные элементы
            for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            
            # Обновляем относительные ссылки
            for tag in soup.find_all(['a', 'link', 'img']):
                attr = 'href' if tag.name in ['a', 'link'] else 'src'
                if tag.has_attr(attr):
                    tag[attr] = urljoin(url, tag[attr])
            
            # Конвертируем в Markdown
            markdown_content = md(str(soup).replace(url + '#', ""))
            
            # Базовая очистка
            lines = markdown_content.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('!['):  # Убираем изображения
                    cleaned_lines.append(line)
            
            return '\n'.join(cleaned_lines)
            
        except Exception as e:
            # Проверяем, похоже ли на рейт-лимит
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['rate limit', 'too many requests', 'throttle']):
                print(f"🚫 Возможный рейт-лимит при загрузке {url}: {e}")
                await asyncio.sleep(0.5)
            else:
                print(f"❌ Ошибка при обработке {url}: {e}")
            return ""

    async def search(self, query: str) -> Dict[str, str]:
        """Выполняет поиск по запросу через Searx и возвращает словарь URL -> содержимое в Markdown.
        Если ничего не найдено, возвращает {'llm_fallback': True} для генерации ответа LLM."""
        timer.reset()
        
        with timer.measure("Кеширование"):
            # Создаем хеш для кеширования
            md5_hash = hashlib.new('md5')
            md5_hash.update(query.encode())
            cache_key = f"searx_search_{md5_hash.hexdigest()}"
            
            # Проверяем кеш
            cached = self.redis_client.get(cache_key)
            if cached is not None:
                try:
                    cached_data = json.loads(cached)
                    print(f"✓ Найден кеш для запроса: {query}")
                    print(timer.get_summary_line())
                    return cached_data
                except:
                    pass

        try:
            print(f"🔍 Выполняем поиск Searx для: {query}")
            with timer.measure("Поиск Searx"):
                results = await self.searx.search(query, num_results=self.ref_cnt)
            
            if not results:
                print("❌ Результаты поиска не найдены")
                return {"llm_fallback": True}
            
            with timer.measure("Фильтрация результатов"):
                # Извлекаем ссылки и сортируем по релевантности
                links_with_scores = []
                for i, result in enumerate(results):
                    if 'url' in result:
                        # Простая оценка релевантности: позиция в поиске (меньше = лучше)
                        score = i
                        # Бонус за наличие ключевых слов в заголовке
                        if 'title' in result:
                            title_lower = result['title'].lower()
                            query_words = query.lower().split()
                            title_bonus = sum(1 for word in query_words if word in title_lower)
                            score -= title_bonus * TITLE_KEYWORD_BONUS  # Снижаем оценку (лучше)
                        
                        links_with_scores.append((result['url'], score))
                
                # Сортируем по оценке и берем лучшие
                links_with_scores.sort(key=lambda x: x[1])
                links = [link for link, _ in links_with_scores]
                
                # Фильтруем по черному списку
                filtered_links = filter_links_by_blacklist(links, self.blacklist)
                if filtered_links:
                    links = filtered_links
                # Ограничиваем количество ссылок
                links = links[:self.ref_cnt]
                
            print(f"📊 Найдено {len(links)} ссылок для обработки")
            with timer.measure("Загрузка страниц"):
                url_md_dict = {}
                semaphore = asyncio.Semaphore(4)  # максимум 4 одновременных запроса
                async def fetch_and_store(url):
                    async with semaphore:
                        md_content = await self.html_to_md(url)
                        if isinstance(md_content, str) and len(md_content.strip()) > MIN_CONTENT_LENGTH:
                            url_md_dict[url] = md_content
                await asyncio.gather(*(fetch_and_store(url) for url in links))
            print(f"✅ Успешно обработано {len(url_md_dict)} страниц")
            with timer.measure("Сохранение в кеш"):
                # Кешируем результат на 1 час
                if url_md_dict:
                    self.redis_client.setex(cache_key, 3600, json.dumps(url_md_dict))
            
            # Выводим итоговую таблицу времени
            print("\n" + timer.get_summary_table())
            
            if not url_md_dict:
                print("❌ Не удалось получить содержимое ни одной страницы, fallback на LLM")
                return {"llm_fallback": True}
            return url_md_dict
            
        except Exception as e:
            print(f"❌ Ошибка при поиске: {e}")
            return {"llm_fallback": True}

    async def close(self):
        """Закрывает HTTP клиент."""
        await self.client.aclose()
