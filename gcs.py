import json
import hashlib
import asyncio
import httpx
import redis
from bs4 import BeautifulSoup
from typing import Dict, List, Set
from markdownify import markdownify as md
from urllib.parse import urljoin, urlparse


def first_word_with_number(text):
    words = text.split()
    for word in words:
        if any(char.isdigit() for char in word):
            return word
    return None


def filter_links_by_blacklist(links: List[str], blacklist: Set[str]) -> List[str]:
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


class GoogleGCS(object):
    def __init__(self, redis_client: redis.Redis, key, id, ref_cnt, gcs_endpoint, timeout, blacklist):
        self.params = {
            "key": key,
            "cx": id,
            "q": None,
            "num": ref_cnt,
            "fileType": "html",
            "cr": "countryRU",
            "gl": "ru",
            "hl": "ru",
            "lr": "lang_ru",
        }
        self.client = httpx.AsyncClient(headers={'User-Agent': 'Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.6998.166 Safari/537.36'})
        self.endpoint = gcs_endpoint
        self.timeout = timeout
        self.redis_client = redis_client
        self.blacklist = blacklist


    async def html_to_md(self, url: str) -> str:
        try:
            response = await self.client.get(url=url, timeout=self.timeout)
            if not response.is_success:
                return ""
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            for tag in soup.find_all(['a', 'link', 'script', 'img']):
                attr = 'href' if tag.name in ['a', 'link'] else 'src'
                if tag.has_attr(attr):
                    tag[attr] = urljoin(url, tag[attr])
            return md(str(soup).replace(url + '#', ""))
        except Exception as e:
            return ""


    async def search(self, query: str) -> Dict[str, str]:
        md5_hash = hashlib.new('md5')
        md5_hash.update(query.encode())
        hash = md5_hash.hexdigest()
        cached = self.redis_client.get(hash)
        if cached is None:
            params = self.params
            params["q"] = query
            exact = first_word_with_number(query)
            if exact is not None:
                params["exactTerms"] = exact
            response = await self.client.get(url=self.endpoint, params=params, timeout=self.timeout)
            if not response.is_success:
                return dict()
            json_content = response.json()
            self.redis_client.set(hash, json.dumps(json_content))
        else:
            json_content = json.loads(cached)
        links = [i["link"] for i in json_content["items"]]
        filtered = filter_links_by_blacklist(links, self.blacklist)
        if len(filtered) > 0:
            links = filtered
        md_results = await asyncio.gather(*(self.html_to_md(url) for url in links))
        url_md_dict = {url: md_content for url, md_content in zip(links, md_results) if len(md_content) > 0}
        return url_md_dict
