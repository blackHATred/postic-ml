"""Модели данных."""
from typing import List
import re


class Chunk:
    """Класс для представления текстового блока с возможностью содержания изображений."""
    
    def __init__(self, s: str, img_url: str = None, img_pos: int = None, begin: int = None, end: int = None):
        self.s = s
        self.i = 0
        self.img = img_url
        self.img_pos = img_pos
        if begin is None and end is None:
            self.begin = 0
            self.end = len(s)
        else:
            self.begin = begin
            self.end = end

    def if_img_for_emb_view(self):
        """Возвращает текст и изображение для векторного представления."""
        if self.img is None:
            return [self.s.strip()], None
        right_before = self.s[self.img_pos:].find('](') + self.img_pos + 2
        right_after = self.img_pos + len(self.img) - 1
        text = (self.s[:right_before] + "<image>" + self.s[right_after:]).strip()
        image = self.s[right_before:right_after].split(" ")[0]
        return [text], [image]

    def split_by_img(self) -> List:
        """Разделяет текст по изображениям."""
        regex = r'!\[[^\]]*\]\((https?://[^\s)]+?\.(?:a?png|jpe?g|jfif|pjpeg|pjp|webp|gif|avif|bmp|tiff?|ico|cur))(?:\s+["\'][^"\']*["\'])?\)'
        matches = list(re.finditer(regex, self.s, re.IGNORECASE))
        if not matches:
            return [self]
        
        chunks = []
        start = 0
        pref_repl_end = 0
        link_start = matches[0].start()
        link_end = matches[0].end()
        img_url = None
        img_pos = None
        
        for i in range(len(matches) - 1):
            end = matches[i + 1].start()
            content = " " * (pref_repl_end - start) + self.s[pref_repl_end:end]
            img_url = self.s[link_start:link_end]
            img_pos = link_start - start
            chunks.append(Chunk(content, img_url, img_pos, self.begin + start, self.begin + end))
            img_url = None
            img_pos = None
            start = matches[i].start()
            pref_repl_end = matches[i].end()
            link_start = matches[i+1].start()
            link_end = matches[i+1].end()
        
        content = self.s[start:]
        img_url = self.s[link_start:link_end]
        img_pos = link_start - start
        chunks.append(Chunk(content, img_url, img_pos, self.begin + start, self.end))
        return chunks

    def split_by_const(self, max_len: int) -> List:
        """Разделяет текст на блоки фиксированного размера с перекрытием."""
        end = len(self.s)
        if max_len >= end:
            return [self]
        
        overlap_len = max_len // 2
        res = []
        start = 0
        
        while start < len(self.s):
            end = min(start + max_len, len(self.s))
            img_url = None
            img_pos = None
            
            if self.img is not None and start <= self.img_pos < end:
                img_url = self.img
                img_pos = self.img_pos - start
            
            res.append(Chunk(self.s[start:end], img_url, img_pos, self.begin + start, self.begin + end))
            
            if end == len(self.s):
                break
            start = end - overlap_len
        
        return res
