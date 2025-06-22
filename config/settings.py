"""Конфигурационные настройки приложения."""
import os

# Настройки сервера
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 8000))

# Настройки Qdrant
QDRANT_HOST = os.environ.get("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

# Настройки Redis
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

# Настройки Ollama
NUM_CTX = int(os.environ.get("NUM_CTX", 8192))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "bge-m3:567m")
OLLAMA_EMBEDDING_MODEL_DIM = int(os.environ.get("OLLAMA_EMBEDDING_MODEL_DIM", 1024))
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", 60))
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "604800s") # 7 дней в секундах
OLLAMA_PRELOAD_MODELS = True  # Предзагружать модели при старте приложения
TEMP = 0.7

# Настройки алгоритмов
MAGIC_COEF = 1
SEARCH_K_COEF = 1.5  # Было 0.65, увеличено для расширения выборки
START_DIVIDE = 1536
OVERALL_CHUNK_COUNT_LIM = 128
DEFAULT_TIMEOUT = 5
REFERENCE_COUNT = 3

# Черный список доменов
DOMAINS_BLACKLIST = {
    ".otzovik.com", "otzovik.com", 
    ".yaplakal.com", "yaplakal.com", 
    ".musavat.ru", "musavat.ru", 
    ".ridlife.ru", "ridlife.ru"
}

# Строковые константы для ответов
STR_NO_ANSWER = "SKIP"
STR_PASS = "REPORT"

# Расширенные настройки поиска DuckDuckGo
SEARCH_TIME_RANGE = 'y'  # 'y' = последний год, 'm' = месяц, 'w' = неделя, 'd' = день
SEARCH_MULTIPLIER = 2  # Множитель для получения большего количества результатов для фильтрации
SEARCH_REGION = 'ru-ru'  # Регион поиска
SEARCH_SAFESEARCH = 'moderate'  # Уровень безопасного поиска

# Настройки ранжирования результатов
TITLE_KEYWORD_BONUS = 0.8  # Бонус за наличие ключевых слов в заголовке
MIN_CONTENT_LENGTH = 1000  # Минимальная длина контента для включения в результат

# Настройки умного поиска и контекста
MAX_SEARCH_QUERIES = 3  # Максимальное количество поисковых запросов
QUERY_EXTRACTION_TEMPERATURE = 0.1  # Температура для извлечения запросов
ENABLE_TIME_CONTEXT = True  # Включить контекст времени в промпты
SEARCH_RESULT_DEDUPLICATION = True  # Удалять дубликаты результатов поиска
CONTEXT_RELEVANCE_THRESHOLD = 0.7  # Порог релевантности для объединения запросов
TIMEZONE = "UTC+3 (Moscow Time)"  # Часовой пояс для контекста времени
