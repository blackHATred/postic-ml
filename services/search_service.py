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


async def index_one(hash_name, client, prev_last_id_chunk_count, url, chunks):
    """Индексирует один документ."""
    prev_last_id, chunk_count = prev_last_id_chunk_count
    if chunk_count > OVERALL_CHUNK_COUNT_LIM:
        return
    
    points = []
    for chunk in chunks:
        texts, images = chunk.if_img_for_emb_view()
        images_ = None
        vector = await get_vector(texts, images_)
        payload = {
            "source": url,
            "text": chunk.s,
            "begin": chunk.begin,
            "end": chunk.end,
        }
        if images is not None:
            payload["img_url"] = images[0]
        
        id = prev_last_id + chunk.i
        points.append(PointStruct(id=id, vector=vector, payload=payload))

    client.upsert(
        collection_name=hash_name,
        points=points
    )


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
        lens = dict()
        for url, md_content in url_md_dict.items():
            url_md_dict[url] = to_chunks(md_content)
            lens[url] = len(url_md_dict[url])
        
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
        
        await asyncio.gather(*(
            index_one(hash_name, client, prev_last_ids_dict[url], url, chunks) 
            for url, chunks in url_md_dict.items()
        ))
        
        return hash_name, chunk_count_pred
    except Exception as e:
        print(f"Исключение во время индексации: {e}")
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


async def get_relevant_documents(client: QdrantClient, collection_name: str, query: str, search_k: int):
    """Получает релевантные документы по запросу."""
    query_vector = await get_vector([query], None)
    top_k_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=search_k,
        with_payload=True,
    )
    neighbors = retrieve_neighbors(client, collection_name, top_k_results)
    all_results = top_k_results + neighbors
    scores, texts, images = combine_results(all_results)
    return scores, texts, images
