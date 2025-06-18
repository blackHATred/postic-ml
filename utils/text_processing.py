"""Вспомогательные функции для обработки текста."""
import re
from typing import List

DEBUG_LOGS = False  # Управляет подробным выводом логов в combine_results и to_chunks

# Локальный импорт для избежания циклических зависимостей
def get_chunk_class():
    from models.chunk import Chunk
    return Chunk


def flatten(xss):
    """Преобразует вложенный список в плоский."""
    return [x for xs in xss for x in xs]


def delete_all_links(text: str) -> str:
    """Удаляет все ссылки из текста."""
    pattern = r'!?\[[^\]\[\)\(]*\]\([^\)\(\]\[]*\)'
    return re.sub(pattern, '', text)


def to_chunks(md_content: str) -> List:
    """Преобразует markdown контент в список блоков."""
    from config.settings import START_DIVIDE
    Chunk = get_chunk_class()
    result = Chunk(md_content).split_by_img()
    if DEBUG_LOGS:
        print(f"[to_chunks] После split_by_img: {len(result)} чанков")
        for idx, ch in enumerate(result):
            print(f"[to_chunks] chunk[{idx}].img = {ch.img}")
    for i in range(len(result)):
        if len(result[i].s) > START_DIVIDE:
            result[i] = result[i].split_by_const(START_DIVIDE)
        else:
            result[i] = [result[i]]
    res = flatten(result)
    result = []
    for i in res:
        if not result or i.end > result[-1].end:
            if result and i.begin <= result[-1].begin:
                result[-1] = i
            else:
                result.append(i)
    if DEBUG_LOGS:
        print(f"[to_chunks] После split_by_const: {len(result)} чанков")
        for idx, ch in enumerate(result):
            print(f"[to_chunks] chunk[{idx}].img = {ch.img}")
    result_len = len(result)
    min_len = len(result[0].s)
    max_len = len(result[0].s)
    for i in range(result_len):
        result[i].i = i
        i_len = len(result[i].s)
        if i_len < min_len:
            min_len = i_len
        elif i_len > max_len:
            max_len = i_len
    return result


def combine_results(results) -> List[str]:
    """Объединяет результаты поиска в связные тексты."""
    from config.settings import MAGIC_COEF
    prev_id = None
    concat = []
    scores = []
    images = []
    chunks_cnts = []
    begin = None
    sorted_results = sorted(results, key=lambda x: x.id)
    for i in sorted_results:
        if i.id - 1 != prev_id:
            concat.append("")
            begin = i.payload["begin"]
            scores.append(0)
            images.append(dict())
            chunks_cnts.append(0)
        if hasattr(i, 'score') and i.score > scores[-1]:
            scores[-1] = i.score
        concat[-1] += i.payload["text"][begin - i.payload["begin"]:]
        chunks_cnts[-1] += 1
        if "img_url" in i.payload:
            images[-1][i.id] = i.payload["img_url"]
            if DEBUG_LOGS:
                print(f"[combine_results] img_url найден в payload: {i.payload['img_url']}")
        begin = i.payload["end"]
        prev_id = i.id
    for i in range(len(scores)):
        scores[i] *= (MAGIC_COEF + len(images[i]) / chunks_cnts[i])
    if DEBUG_LOGS:
        print(f"[combine_results] images: {images}")
    if not scores or not concat or not images:
        if DEBUG_LOGS:
            print(f"[combine_results] ВНИМАНИЕ: возвращаем пустые списки! scores={scores}, concat={concat}, images={images}")
        return [], [], []
    try:
        zipped = [list(t) for t in zip(*sorted(zip(scores, concat, images), reverse=True))]
        if len(zipped) != 3:
            if DEBUG_LOGS:
                print(f"[combine_results] ВНИМАНИЕ: результат zip не длины 3! zipped={zipped}")
            return [], [], []
        return zipped
    except Exception as e:
        if DEBUG_LOGS:
            print(f"[combine_results] ОШИБКА: {e}")
        return [], [], []
