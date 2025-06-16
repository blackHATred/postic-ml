"""Тестирование DuckDuckGo поиска."""
import asyncio
import redis
from services.duckduckgo_search import DuckDuckGoSearch
from config.settings import REDIS_HOST, REDIS_PORT, REFERENCE_COUNT, DEFAULT_TIMEOUT, DOMAINS_BLACKLIST


async def test_duckduckgo_search():
    """Тестирует работу DuckDuckGo поиска."""
    # Подключаемся к Redis
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    
    # Создаем поисковый сервис
    searcher = DuckDuckGoSearch(
        redis_client=redis_client,
        ref_cnt=3,  # Для теста берем только 3 результата
        timeout=DEFAULT_TIMEOUT,
        blacklist=DOMAINS_BLACKLIST
    )
    
    try:
        print("Тестируем DuckDuckGo поиск...")
        
        # Выполняем тестовый поиск
        query = "Python programming tutorial"
        results = await searcher.search(query)
        
        print(f"\nРезультаты поиска для '{query}':")
        print(f"Найдено {len(results)} страниц")
        
        for url, content in results.items():
            print(f"\nURL: {url}")
            print(f"Содержимое (первые 200 символов): {content[:200]}...")
        
        if results:
            print("\n✅ DuckDuckGo поиск работает корректно!")
        else:
            print("\n❌ Поиск не вернул результатов")
    
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
    
    finally:
        await searcher.close()


if __name__ == "__main__":
    asyncio.run(test_duckduckgo_search())
