#!/usr/bin/env python3
"""Скрипт для предварительной загрузки моделей."""

import os
import sys
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

def download_models():
    """Загружает все необходимые модели и сохраняет их локально."""
    
    # Создаем директории для моделей
    models_dir = "/app/models_cache"
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        print("Загрузка модели анализа тональности...")
        # Модель анализа тональности
        sentiment_model_name = "blanchefort/rubert-base-cased-sentiment"
        sentiment_cache_dir = f"{models_dir}/sentiment"
        
        # Загружаем токенизатор и модель отдельно для лучшего контроля кэша
        tokenizer = AutoTokenizer.from_pretrained(
            sentiment_model_name,
            cache_dir=sentiment_cache_dir
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            sentiment_model_name,
            cache_dir=sentiment_cache_dir
        )
        
        # Создаем pipeline для проверки работоспособности
        sentiment_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device="cpu"
        )
        
        # Тестируем модель
        test_result = sentiment_pipeline("Это тестовое сообщение")
        print(f"✓ Модель анализа тональности загружена и протестирована: {test_result[0]['label']}")
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели анализа тональности: {e}")
        sys.exit(1)
    
    try:
        print("Загрузка модели для векторного представления...")
        # Модель для векторного представления
        seq_model_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"
        seq_cache_dir = f"{models_dir}/sentence_transformer"
        
        seq_model = SentenceTransformer(
            seq_model_name,
            cache_folder=seq_cache_dir,
            device="cpu"
        )
        
        # Тестируем модель
        test_embedding = seq_model.encode("Это тестовое сообщение")
        print(f"✓ Модель для векторного представления загружена и протестирована: {len(test_embedding)} измерений")
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели векторного представления: {e}")
        sys.exit(1)
    
    print("Все модели успешно загружены и закэшированы!")
    print(f"Размер кэша моделей:")
    
    # Выводим размер каждой директории
    total_size = 0
    for root, dirs, files in os.walk(models_dir):
        dir_size = sum(os.path.getsize(os.path.join(root, file)) for file in files)
        if dir_size > 0:
            size_mb = dir_size / 1024 / 1024
            print(f"  {root}: {size_mb:.1f} MB")
            total_size += size_mb
    
    print(f"Общий размер кэша: {total_size:.1f} MB")

if __name__ == "__main__":
    download_models()
