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
TEMP = 1.0

# Настройки алгоритмов
MAGIC_COEF = 1
SEARCH_K_COEF = 0.65
START_DIVIDE = 2048
DEFAULT_TIMEOUT = 5
REFERENCE_COUNT = 10
OVERALL_CHUNK_COUNT_LIM = 128

# Настройки поиска
DEFAULT_TIMEOUT = 5
REFERENCE_COUNT = 10
OVERALL_CHUNK_COUNT_LIM = 128

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
