# 🤖 Postic ML — AI-Powered Social Media Management API

Мощный API для автоматизации управления контентом в социальных сетях с использованием искусственного интеллекта.

---

## 🚀 Особенности
- Генерация и анализ контента с помощью LLM (Ollama)
- Векторный поиск (Qdrant) и эмбеддинги
- Асинхронная архитектура на FastAPI
- Redis-кеширование
- DuckDuckGo-поиск
- Модульная структура и расширяемость

---

## 📦 Установка и запуск

**Требования:** Python 3.12+, Redis, Qdrant, Ollama с нужными моделями

```bash
# Клонирование репозитория
 git clone <repository-url>
 cd postic-ml

# Установка зависимостей
pip install -r requirements.txt

# Настройка переменных окружения
cp example.env .env
# Отредактируйте .env под свои настройки

# Запуск сервера
python main.py
```

---

## 🔌 API Документация

### Base URL: `http://localhost:8000`

#### `POST /publication`
Генерирует пост для соцсетей на основе поискового запроса

**Request:**
```json
{
  "query": "составь пост про здоровое питание"
}
```
**Response:**
```json
{
  "text": "Решил поделиться своим опытом здорового питания...",
  "images": [
    "https://example.com/image1.jpg"
  ]
}
```

#### `POST /sentiment`
Анализ тональности комментария

**Request:**
```json
{"comment": "Отличный продукт!"}
```
**Response:**
```json
{"label": "POSITIVE", "score": 0.98}
```

#### `POST /ticket_synt` / `POST /ticket_llm`
Определяет необходимость поддержки (ML/LLM)

**Request:**
```json
{"comment": "У меня не работает функция загрузки"}
```
**Response:**
```json
{"support_needed": true}
```

#### `POST /sum`
Суммаризация комментариев

**Request:**
```json
{"comments": "Пользователь 1: Отлично!\nПользователь 2: Спасибо!"}
```
**Response:**
```json
{"response": "Краткое содержание: пользователи положительно оценили контент"}
```

#### `POST /fix`
Исправляет ошибки в тексте

**Request:**
```json
{"text": "Превет! Как дила?"}
```
**Response:**
```json
{"response": "Привет! Как дела?"}
```

#### `POST /ans`
Генерирует ответы на комментарии

**Request:**
```json
{"comment": "Как подключить наушники?", "style": "дружелюбном"}
```
**Response:**
```json
{"no_answer": false, "support_needed": false, "answer_0": "Включите Bluetooth..."}
```

#### `GET /health`
Проверка работоспособности сервиса

---

## 📈 Примеры использования

**Python:**
```python
import httpx, asyncio
async def generate_post():
    async with httpx.AsyncClient() as client:
        r = await client.post("http://localhost:8000/publication", json={"query": "польза йоги"})
        return r.json()
print(asyncio.run(generate_post()))
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/publication" -H "Content-Type: application/json" -d '{"query": "рецепт борща"}'
curl -X POST "http://localhost:8000/sentiment" -H "Content-Type: application/json" -d '{"comment": "Супер качество!"}'
```

---

## 🏗️ Архитектура

```
postic-ml/
├── api/                # API эндпоинты
├── config/             # Конфигурация
├── core/               # Инициализация
├── models/             # Pydantic и ML-модели
├── services/           # Бизнес-логика
├── utils/              # Утилиты
└── main.py             # Точка входа
```

**Технологии:** FastAPI, Ollama, Qdrant, Redis, DuckDuckGo, PyTorch, httpx, Pydantic

---

## ⚙️ Конфигурация

Пример .env:
```env
HOST=127.0.0.1
PORT=8000
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
OLLAMA_EMBEDDING_MODEL=bge-m3:567m
OLLAMA_TIMEOUT=60
REFERENCE_COUNT=10
DEFAULT_TIMEOUT=5
```


---

**Сделано с ❤️ в рамках VK Education для проекта Postic**