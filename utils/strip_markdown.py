import re

def strip_markdown_and_links(text: str) -> str:
    """
    Удаляет markdown-разметку (заголовки, жирный, курсив, списки, code, цитаты) и все ссылки из текста.
    Оставляет только чистый текст.
    """
    # Удалить markdown-ссылки и изображения
    text = re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', text)  # картинки ![alt](url)
    text = re.sub(r'\[[^\]]*\]\([^\)]*\)', '', text)   # ссылки [text](url)
    # Удалить заголовки
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Удалить жирный и курсив (*, _, **, __)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    # Удалить inline code
    text = re.sub(r'`([^`]*)`', r'\1', text)
    # Удалить блоки кода
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Удалить списки
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Удалить цитаты
    text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)
    # Удалить горизонтальные линии
    text = re.sub(r'^---$', '', text, flags=re.MULTILINE)
    # Удалить лишние пустые строки
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
