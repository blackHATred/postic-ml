"""Утилиты для измерения времени выполнения операций."""
import time
from typing import Dict, List
from contextlib import contextmanager


class TimingTracker:
    """Класс для отслеживания времени выполнения различных операций."""
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
    
    def start(self, operation: str):
        """Начинает отслеживание времени для операции."""
        self.start_times[operation] = time.time()
    
    def stop(self, operation: str):
        """Останавливает отслеживание времени для операции."""
        if operation in self.start_times:
            elapsed = time.time() - self.start_times[operation]
            self.timings[operation] = elapsed
            del self.start_times[operation]
            return elapsed
        return 0
    
    @contextmanager
    def measure(self, operation: str):
        """Контекстный менеджер для измерения времени операции."""
        self.start(operation)
        try:
            yield
        finally:
            self.stop(operation)
    
    def get_summary_table(self) -> str:
        """Возвращает красивую таблицу с результатами измерений."""
        if not self.timings:
            return "Нет данных о времени выполнения"
        
        # Заголовок таблицы
        lines = []
        lines.append("┌─────────────────────────────┬──────────────┐")
        lines.append("│ Операция                    │ Время (сек)  │")
        lines.append("├─────────────────────────────┼──────────────┤")
        
        # Строки с данными
        total_time = 0
        for operation, timing in self.timings.items():
            operation_str = operation[:27] + "..." if len(operation) > 30 else operation
            timing_str = f"{timing:.3f}"
            lines.append(f"│ {operation_str:<27} │ {timing_str:>10}   │")
            total_time += timing
        
        # Общее время
        lines.append("├─────────────────────────────┼──────────────┤")
        lines.append(f"│ {'ОБЩЕЕ ВРЕМЯ':<27} │ {total_time:.3f}   │")
        lines.append("└─────────────────────────────┴──────────────┘")
        
        return "\n".join(lines)
    
    def get_summary_line(self) -> str:
        """Возвращает краткую сводку в одной строке."""
        if not self.timings:
            return "Нет данных о времени выполнения"
        
        parts = []
        total_time = 0
        for operation, timing in self.timings.items():
            parts.append(f"{operation}: {timing:.2f}с")
            total_time += timing
        
        return f"[ВРЕМЯ] {' | '.join(parts)} | ОБЩЕЕ: {total_time:.2f}с"
    
    def reset(self):
        """Сбрасывает все измерения."""
        self.timings.clear()
        self.start_times.clear()


# Глобальный экземпляр для использования во всем приложении
timer = TimingTracker()
