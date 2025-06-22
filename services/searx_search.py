import httpx
from typing import List, Dict, Optional
import itertools
import traceback

SEARX_INSTANCES = [
    # "https://searx.stream/",
    # "https://searx.tiekoetter.com/",
    # "https://search.hbubli.cc/",
    # "https://search.rhscz.eu/",
    # "https://priv.au/",
    # "https://searx.perennialte.ch/",
    # "https://opnxng.com/",
    # "https://searx.rhscz.eu/",
    # "https://search.ipv6s.net/",
    # "https://searx.foobar.vip/",
    # "https://search.rowie.at/",
    # "https://search.sapti.me/",
    # "https://searx.dresden.network/",
    # "https://searxng.site/",
    # "https://searxng.hweeren.com/",
    # "https://search.mdosch.de/",
    # "https://search.leptons.xyz/",
    # "https://find.xenorio.xyz/",
    # "https://kantan.cat/",
    # "https://www.gruble.de/",
    # "https://searxng.f24o.zip/",
    # "https://search.catboy.house/",
    # "https://baresearch.org/",
    # "https://searx.tuxcloud.net/",
    # "https://search.einfachzocken.eu/",
    # "https://search.nordh.tech/",
    # "https://searx.ox2.fr/",
    # "https://searxng.deliberate.world/",
    # "https://search.ononoki.org/",
    # "https://search.federicociro.com/",
    # "https://copp.gg/",
    # "https://searx.foss.family/",
    # "https://search.citw.lgbt/",
    # "https://search.im-in.space/",
    # "https://search.080609.xyz/",
    # "https://searxng.shreven.org/",
    # "https://searx.namejeff.xyz/",
    # "https://search.nerdvpn.de/"
    "http://searxng.postic-ml:8080/",  # Внутренний адрес k8s-инстанса
]

class SearxSearch:
    def __init__(self, instances: Optional[List[str]] = None, timeout: float = 2):
        self.instances = instances or SEARX_INSTANCES
        self.timeout = timeout
        self._cycle = itertools.cycle(self.instances)

    async def search(self, query: str, num_results: int = 5) -> List[Dict]:
        headers = {"User-Agent": "Mozilla/5.0"}
        params = {
            "q": query,
            "format": "json",
            "language": "ru-RU",  # Язык поиска
            "safesearch": 1,  # Безопасный поиск
            "count": num_results
        }
        for _ in range(len(self.instances)):
            instance = next(self._cycle)
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.get(f"{instance}/search", params=params, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        results = data.get("results", [])
                        if not results:
                            print(f"[DEBUG] Empty results! Full response: {resp.text}")
                        return results
                    else:
                        print(f"Searx instance {instance} returned {resp.status_code}. Response: {resp.text}")
            except Exception as e:
                print(f"Searx error with {instance}: {e}\n{traceback.format_exc()}")
        return []

    async def search_images(self, query: str, num_results: int = 5) -> List[Dict]:
        """Специальный поиск изображений через SearX."""
        from config.settings import IMAGE_SEARCH_ENGINES
        
        headers = {"User-Agent": "Mozilla/5.0"}
        params = {
            "q": query,
            "format": "json",
            "categories": "images",  # Указываем категорию изображений
            "count": num_results,
            "language": "ru-RU",  # Язык поиска
            "safesearch": 1,  # Безопасный поиск
            "time_range": "",  # Без ограничений по времени для изображений
            # "engines": IMAGE_SEARCH_ENGINES  # Движки для изображений
        }
        
        for _ in range(len(self.instances)):
            instance = next(self._cycle)
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.get(f"{instance}/search", params=params, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        results = data.get("results", [])
                        
                        # Фильтруем результаты, оставляя только прямые ссылки на изображения
                        image_results = []
                        for result in results:
                            img_url = (result.get("img_src") or 
                                     result.get("thumbnail") or 
                                     result.get("content") or 
                                     result.get("url"))
                            
                            if img_url and self._is_direct_image_url(img_url):
                                image_results.append({
                                    "url": img_url,
                                    "title": result.get("title", ""),
                                    "thumbnail": result.get("thumbnail", ""),
                                    "img_src": img_url
                                })
                        
                        if not image_results:
                            print(f"[DEBUG] No image results! Full response: {data}")
                        return image_results
                    else:
                        print(f"Searx images instance {instance} returned {resp.status_code}")
            except Exception as e:
                print(f"Searx images error with {instance}: {e}")
        return []

    def _is_direct_image_url(self, url: str) -> bool:
        """Проверяет, является ли URL прямой ссылкой на изображение."""
        if not url or not isinstance(url, str):
            return False
        
        url_lower = url.lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.ico']
        
        # Проверяем расширение файла
        for ext in image_extensions:
            if url_lower.endswith(ext):
                return True
        
        # Проверяем наличие параметров изображений в URL
        image_params = ['format=jpg', 'format=jpeg', 'format=png', 'format=webp', 
                       '.jpg?', '.jpeg?', '.png?', '.webp?', '.gif?']
        for param in image_params:
            if param in url_lower:
                return True
        
        # Проверяем домены изображений
        image_domains = ['images.', 'img.', 'media.', 'cdn.', 'static.', 'assets.']
        for domain in image_domains:
            if domain in url_lower:
                return True
                
        return False
