"""Инициализация клиентов и моделей."""
import os
import redis
import torch
from qdrant_client import QdrantClient
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from config.settings import (
    QDRANT_HOST, QDRANT_PORT, 
    REDIS_HOST, REDIS_PORT,
    REFERENCE_COUNT, DEFAULT_TIMEOUT, DOMAINS_BLACKLIST,
    OLLAMA_PRELOAD_MODELS
)
from models.classifier import TwoLayerClassifier
from services.duckduckgo_search import DuckDuckGoSearch


async def preload_ollama_models():
    """Предзагружает модели Ollama если включена соответствующая настройка."""
    if OLLAMA_PRELOAD_MODELS:
        from services.llm_service import preload_models
        await preload_models()
    else:
        print("⏭️ Предзагрузка моделей Ollama отключена")


def init_clients():
    """Инициализирует клиенты для внешних сервисов."""
    # Qdrant клиент
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    if not client:
        print("Подключение к qdrant - FAIL")
        exit()
    print("Подключение к qdrant - OK")
    
    # Redis клиент
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        if redis_client.ping():
            print("Подключение к redis - OK")
    except redis.ConnectionError as e:
        print("Подключение к redis - FAIL:", e)
        exit()
      # DuckDuckGo поисковый клиент
    searcher = DuckDuckGoSearch(
        redis_client, REFERENCE_COUNT, 
        DEFAULT_TIMEOUT, DOMAINS_BLACKLIST
    )
    
    return client, redis_client, searcher


def init_models():
    """Инициализирует ML модели."""
    # Принудительно используем CPU, если установлена переменная окружения
    force_cpu = os.getenv('FORCE_CPU', 'true').lower() == 'true'
    device = 'cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Используемое устройство для моделей: {device}")
    
    # Пути к закэшированным моделям
    models_cache_dir = "/app/models_cache"
    sentiment_cache_dir = f"{models_cache_dir}/sentiment"
    seq_cache_dir = f"{models_cache_dir}/sentence_transformer"
    
    # Проверяем, существуют ли локальные модели
    if os.path.exists(sentiment_cache_dir) and os.path.exists(seq_cache_dir):
        print("Загружаем модели из локального кэша...")
        
        # Модель анализа тональности из кэша
        sentiment_model = pipeline(
            model="blanchefort/rubert-base-cased-sentiment",
            device_map="cpu" if force_cpu else "auto",
            model_kwargs={"cache_dir": sentiment_cache_dir, "local_files_only": True}
        )
        
        # Модель для векторного представления из кэша
        seq_model = SentenceTransformer(
            'sentence-transformers/distiluse-base-multilingual-cased-v1',
            device=device,
            cache_folder=seq_cache_dir
        )
        print("✓ Модели загружены из локального кэша")
    else:
        print("Локальный кэш не найден, загружаем модели из интернета...")
        
        # Модель анализа тональности
        sentiment_model = pipeline(
            model="blanchefort/rubert-base-cased-sentiment",
            device_map="cpu" if force_cpu else "auto"
        )
        
        # Модель для векторного представления
        seq_model = SentenceTransformer(
            'sentence-transformers/distiluse-base-multilingual-cased-v1',
            device=device
        )
    
    # Модель классификации тикетов
    ticket_model = TwoLayerClassifier()
    ticket_model.load_state_dict(torch.load(
        "synt_ticket_model_weights.pth",
        map_location=device,  # Загружаем на правильное устройство
        weights_only=True
    ))
    ticket_model.to(device)  # Перемещаем модель на устройство
    ticket_model.eval()
    
    return sentiment_model, seq_model, ticket_model
