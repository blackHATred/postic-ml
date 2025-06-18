"""Сервис для индексации и поиска документов."""
import time
import math
import hashlib
import asyncio
from typing import List, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http.models import HnswConfigDiff, PointStruct

from config.settings import (
    OLLAMA_EMBEDDING_MODEL_DIM, 
    OVERALL_CHUNK_COUNT_LIM, 
    SEARCH_K_COEF
)
from services.embedding_service import get_vector
from utils.text_processing import to_chunks, combine_results
from utils.timing import timer


async def index_one(hash_name, client, prev_last_id_chunk_count, url, chunks):
    """Индексирует один документ с батчингом эмбеддинга."""
    import time
    prev_last_id, chunk_count = prev_last_id_chunk_count
    if chunk_count > OVERALL_CHUNK_COUNT_LIM:
        print(f"[index_one] Пропуск url (слишком много чанков): {url}")
        return
    points = []
    t0 = time.time()
    batch_size = 4
    chunk_batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
    emb_idx = 0
    for batch in chunk_batches:
        t_emb_start = time.time()
        texts_list = []
        for chunk in batch:
            texts, _ = chunk.if_img_for_emb_view()
            if isinstance(texts, list):
                texts_list.append(texts[0])
            else:
                texts_list.append(texts)
        vectors = await get_vector(texts_list, None)
        t_emb_end = time.time()
        print(f"[index_one] {url} | Батч {emb_idx}: эмбеддинг {len(batch)} чанков занял {t_emb_end - t_emb_start:.3f} сек")
        for i, chunk in enumerate(batch):
            payload = {
                "source": url,
                "text": chunk.s,
                "begin": chunk.begin,
                "end": chunk.end,
            }
            if chunk.img is not None:
                payload["img_url"] = chunk.img
            id = prev_last_id + chunk.i
            points.append(PointStruct(id=id, vector=vectors[i], payload=payload))
        emb_idx += 1
    t_upsert_start = time.time()
    client.upsert(
        collection_name=hash_name,
        points=points
    )
    t_upsert_end = time.time()
    print(f"[index_one] {url} | upsert {len(points)} точек занял {t_upsert_end - t_upsert_start:.3f} сек")
    print(f"[index_one] {url} | всего обработка заняла {t_upsert_end - t0:.3f} сек")


async def index(client: QdrantClient, searcher, query: str):
    """Индексирует документы по запросу."""
    emb_size = OLLAMA_EMBEDDING_MODEL_DIM
    md5_hash = hashlib.new('md5')
    md5_hash.update((query + str(time.time())).encode())
    hash_name = md5_hash.hexdigest()
    
    if not client.collection_exists(collection_name=hash_name):
        client.create_collection(
            collection_name=hash_name,
            vectors_config=VectorParams(
                size=emb_size,
                distance=Distance.COSINE,
                on_disk=False,
                hnsw_config=HnswConfigDiff(ef_construct=100, m=16, on_disk=False)
            ),
            on_disk_payload=False
        )
    else:
        return None, None
    
    try:
        url_md_dict = await searcher.search(query)
        print(f"[index] Поиск завершён. Количество url: {len(url_md_dict)}")
        lens = dict()
        for url, md_content in url_md_dict.items():
            chunks = to_chunks(md_content)
            # Ограничение на максимум 5 чанков на документ
            if len(chunks) > 5:
                chunks = chunks[:5]
            url_md_dict[url] = chunks
            lens[url] = len(url_md_dict[url])
            print(f"[index] URL: {url} | Чанков: {lens[url]} | Длина текста: {len(md_content) if isinstance(md_content, str) else 'N/A'}")
        url_md_dict = dict(sorted(url_md_dict.items(), key=lambda item: len(item[1])))
        prev_last_id = 0
        prev_last_ids_dict = dict()
        chunk_count = 0
        chunk_count_pred = None
        for url, chunks in url_md_dict.items():
            l = len(chunks)
            if chunk_count + l > OVERALL_CHUNK_COUNT_LIM and chunk_count_pred is None:
                chunk_count_pred = chunk_count
            chunk_count += l
            prev_last_ids_dict[url] = [prev_last_id, chunk_count]
            prev_last_id = chunks[-1].i + prev_last_id + 3
            print(f"[index] URL: {url} | prev_last_id: {prev_last_id} | chunk_count: {chunk_count}")
        await asyncio.gather(*(
            index_one(hash_name, client, prev_last_ids_dict[url], url, chunks) 
            for url, chunks in url_md_dict.items()
        ))
        print(f"[index] Индексация завершена. hash_name: {hash_name}, chunk_count: {chunk_count}")
        # Если chunk_count_pred остался None, используем общий chunk_count
        final_chunk_count = chunk_count_pred if chunk_count_pred is not None else chunk_count
        return hash_name, final_chunk_count
    except Exception as e:
        import traceback
        print(f"Исключение во время индексации: {e}\n{traceback.format_exc()}")
        return None, None


def retrieve_neighbors(client: QdrantClient, collection_name: str, results: Any) -> List:
    """Получает соседние документы для улучшения контекста."""
    results = sorted(results, key=lambda x: x.id)
    neighbor_ids = set()
    prev_ci = None
    useless = set()
    
    for result in results:
        center_id = result.id
        neighbor_ids.update([center_id - 2, center_id - 1, center_id + 1, center_id + 2])
        
        if prev_ci is None:
            useless.update([center_id - 2, center_id - 1])
        elif center_id - prev_ci > 2:
            useless.update([prev_ci + 1, prev_ci + 2])
            useless.update([center_id - 2, center_id - 1])
        
        prev_ci = center_id
    
    useless.update([prev_ci + 1, prev_ci + 2])

    existing_ids = {res.id for res in results}
    neighbor_ids -= existing_ids
    neighbor_ids -= useless

    if neighbor_ids:
        return client.retrieve(
            collection_name=collection_name,
            ids=list(neighbor_ids),
            with_payload=True,
        )
    return []


DEBUG_LOGS = False  # Управляет подробным выводом логов в get_relevant_documents

async def get_relevant_documents(client: QdrantClient, collection_name: str, query: str, search_k: int):
    """Получает релевантные документы по запросу."""
    query_vector = (await get_vector([query], None))[0]  # Берём только первый вектор
    top_k_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=search_k,
        with_payload=True,
    )
    if DEBUG_LOGS:
        print(f"[get_relevant_documents] Qdrant вернул {len(top_k_results)} результатов")
        for idx, res in enumerate(top_k_results):
            print(f"[get_relevant_documents] result[{idx}]: id={res.id}, payload={res.payload}")
    neighbors = retrieve_neighbors(client, collection_name, top_k_results)
    if DEBUG_LOGS:
        print(f"[get_relevant_documents] Соседей найдено: {len(neighbors)}")
    all_results = top_k_results + neighbors
    scores, texts, images = combine_results(all_results)
    if DEBUG_LOGS:
        print(f"[get_relevant_documents] combine_results вернул: scores={scores}, texts count={len(texts)}, images={images}")
    if not texts and DEBUG_LOGS:
        print(f"[get_relevant_documents] ВНИМАНИЕ: combine_results вернул пустой список текстов!")
    return scores, texts, images
