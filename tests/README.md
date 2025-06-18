# Tests

Папка для тестов Postic ML приложения.

## Структура

- `__init__.py` - Инициализация пакета тестов
- Здесь будут размещены тесты для различных компонентов приложения

## Запуск тестов

```bash
# Запуск всех тестов
python -m pytest tests/

# Запуск с покрытием
python -m pytest tests/ --cov=.

# Запуск конкретного теста
python -m pytest tests/test_specific.py
```

## Создание тестов

Создавайте тестовые файлы с префиксом `test_` для автоматического обнаружения pytest.

Пример структуры тестового файла:
```python
"""Tests for module_name."""
import pytest
from module_name import function_to_test


def test_function_to_test():
    """Test function description."""
    result = function_to_test()
    assert result == expected_value
```
