"""Утилиты для работы с контекстом времени и даты."""
import os
from datetime import datetime
from typing import Dict, Any


def get_current_context() -> Dict[str, Any]:
    """Получить контекст текущего времени и даты."""
    now = datetime.now()
    
    # Определяем часовой пояс из переменной окружения или используем UTC+3 (Москва)
    timezone = os.environ.get("TIMEZONE", "UTC+3 (Moscow Time)")    
    context = {
        "current_date": now.strftime("%Y-%m-%d"),
        "current_time": now.strftime("%H:%M"),
        "current_datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "day_of_week": get_day_name(now.weekday()),
        "month": get_month_name(now.month),
        "current_month_name": get_month_name(now.month),
        "year": now.year,
        "current_year": str(now.year),
        "hour": now.hour,
        "day": now.day,
        "weekday_number": now.weekday() + 1,  # 1=понедельник, 7=воскресенье
        "is_weekend": now.weekday() >= 5,
        "time_of_day": get_time_of_day(now.hour),
        "season": get_season(now.month),
        "timezone": timezone
    }
    
    return context


def get_day_name(weekday: int) -> str:
    """Получить название дня недели."""
    days = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
    return days[weekday]


def get_month_name(month: int) -> str:
    """Получить название месяца."""
    months = [
        "", "января", "февраля", "марта", "апреля", "мая", "июня",
        "июля", "августа", "сентября", "октября", "ноября", "декабря"
    ]
    return months[month]


def get_time_of_day(hour: int) -> str:
    """Определить время суток."""
    if 5 <= hour < 12:
        return "утро"
    elif 12 <= hour < 17:
        return "день"
    elif 17 <= hour < 22:
        return "вечер"
    else:
        return "ночь"


def get_season(month: int) -> str:
    """Определить время года."""
    if month in [12, 1, 2]:
        return "зима"
    elif month in [3, 4, 5]:
        return "весна"
    elif month in [6, 7, 8]:
        return "лето"
    else:
        return "осень"


def format_context_for_llm(context: Dict[str, Any]) -> str:
    """Форматировать контекст для LLM."""
    weekend_info = "Выходной день" if context['is_weekend'] else "Рабочий день"
    
    return f"""Текущая дата и время: {context['current_datetime']} ({context['timezone']})
Сегодня: {context['day_of_week']}, {context['day']} {context['month']} {context['year']} года
Время суток: {context['time_of_day']}
Сезон: {context['season']}
{weekend_info}"""


def get_date_keywords_for_search(context: Dict[str, Any]) -> str:
    """Получить ключевые слова даты для поиска."""
    keywords = [
        str(context['year']),
        context['month'],
        context['season']
    ]
    
    # Добавляем "сегодня", "вчера", "на этой неделе" для актуальных запросов
    time_keywords = ["сегодня", "сейчас", "текущий", "актуальный", "последние новости"]
    
    return " ".join(keywords + time_keywords)
