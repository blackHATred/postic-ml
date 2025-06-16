"""Сервис для поиска через DuckDuckGo."""
import json
import hashlib
import asyncio
import httpx
import redis
from bs4 import BeautifulSoup
from typing import Dict, List, Set
from markdownify import markdownify as md
from urllib.parse import urljoin, urlparse
from duckduckgo_search import DDGS


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


class DuckDuckGoSearch:
    """Сервис для поиска через DuckDuckGo без ограничений API."""
    
    def __init__(self, redis_client: redis.Redis, ref_cnt: int, timeout: int, blacklist: Set[str]):
        self.redis_client = redis_client
        self.ref_cnt = ref_cnt
        self.timeout = timeout
        self.blacklist = blacklist
        self.client = httpx.AsyncClient(
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )

    async def html_to_md(self, url: str) -> str:
        """Конвертирует HTML страницу в Markdown."""
        try:
            response = await self.client.get(url=url, timeout=self.timeout)
            if not response.is_success:
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
            print(f"Ошибка при обработке {url}: {e}")
            return ""

    async def search(self, query: str) -> Dict[str, str]:
        """Выполняет поиск по запросу и возвращает словарь URL -> содержимое в Markdown."""
        # Создаем хеш для кеширования
        md5_hash = hashlib.new('md5')
        md5_hash.update(query.encode())
        cache_key = f"ddg_search_{md5_hash.hexdigest()}"
        
        # Проверяем кеш
        cached = self.redis_client.get(cache_key)
        if cached is not None:
            try:
                cached_data = json.loads(cached)
                print(f"Найден кеш для запроса: {query}")
                return cached_data
            except:
                pass

        try:
            print(f"Выполняем поиск DuckDuckGo для: {query}")
            
            # Выполняем поиск через DuckDuckGo
            with DDGS() as ddgs:
                # Ищем веб-результаты
                results = list(ddgs.text(
                    query,
                    region='ru-ru',
                    safesearch='moderate',
                    max_results=self.ref_cnt
                ))
            
            if not results:
                print("Результаты поиска не найдены")
                return {}
            
            # Извлекаем ссылки
            links = [result['href'] for result in results if 'href' in result]
            
            # Фильтруем по черному списку
            filtered_links = filter_links_by_blacklist(links, self.blacklist)
            if filtered_links:
                links = filtered_links
            
            print(f"Найдено {len(links)} ссылок для обработки")
            
            # Ограничиваем количество ссылок
            links = links[:self.ref_cnt]
            
            # Получаем содержимое страниц
            md_results = await asyncio.gather(
                *(self.html_to_md(url) for url in links),
                return_exceptions=True
            )
            
            # Создаем словарь результатов
            url_md_dict = {}
            for url, md_content in zip(links, md_results):
                if isinstance(md_content, str) and len(md_content.strip()) > 100:  # Минимальная длина контента
                    url_md_dict[url] = md_content
            
            print(f"Успешно обработано {len(url_md_dict)} страниц")
            
            # Кешируем результат на 1 час
            if url_md_dict:
                self.redis_client.setex(cache_key, 3600, json.dumps(url_md_dict))
            
            return url_md_dict
            
        except Exception as e:
            print(f"Ошибка при поиске: {e}")
            return {}

    async def close(self):
        """Закрывает HTTP клиент."""
        await self.client.aclose()
