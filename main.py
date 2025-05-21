import os
import re
import math
import time
import json
import redis
import torch
import hashlib
import uvicorn
import asyncio
import torch.nn as nn
from typing import List, Any
from ollama import AsyncClient
from transformers import pipeline
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http.models import HnswConfigDiff
from qdrant_client.http.models import PointStruct
from fastapi import FastAPI, HTTPException, Request
from fastapi import FastAPI, HTTPException, Request
from sentence_transformers import SentenceTransformer

from gcs import GoogleGCS


class TwoLayerClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 8000))

QDRANT_HOST = os.environ.get("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

NUM_CTX = int(os.environ.get("NUM_CTX", 8192))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "bge-m3:567m")
OLLAMA_EMBEDDING_MODEL_DIM = int(os.environ.get("OLLAMA_EMBEDDING_MODEL_DIM", 1024))
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", 60))
TEMP = 1.0

MAGIC_COEF = 1
SEARCH_K_COEF = 0.65
START_DIVIDE = 2048
DEFAULT_TIMEOUT = 5
REFERENCE_COUNT = 10
OVERALL_CHUNK_COUNT_LIM = 128
GOOGLE_SEARCH_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"
GOOGLE_GCS_KEY = os.environ.get("GOOGLE_GCS_KEY", "oops")
GOOGLE_GCS_ID = os.environ.get("GOOGLE_GCS_ID", "oops")
DOMAINS_BLACKLIST = {".otzovik.com", "otzovik.com", ".yaplakal.com", "yaplakal.com", ".musavat.ru", "musavat.ru", ".ridlife.ru", "ridlife.ru"}
CLIENT = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
if not CLIENT:
    print("Подключение к qdrant - FAIL")
    exit()
print("Подключение к qdrant - OK")
try:
    REDIS_CLIENT = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    if REDIS_CLIENT.ping():
        print("Подключение к redis - OK")
except redis.ConnectionError as e:
    print("Подключение к redis - FAIL:", e)
    exit()
SEARCHER = GoogleGCS(REDIS_CLIENT, GOOGLE_GCS_KEY, GOOGLE_GCS_ID, REFERENCE_COUNT, GOOGLE_SEARCH_ENDPOINT, DEFAULT_TIMEOUT, DOMAINS_BLACKLIST)

STR_NO_ANSWER = "SKIP"
STR_PASS = "REPORT"
sentiment_model = pipeline(
    model="blanchefort/rubert-base-cased-sentiment",
    device_map="cpu")
# ticket_synt
seq_model = SentenceTransformer(
    'sentence-transformers/distiluse-base-multilingual-cased-v1')
ticket_model = TwoLayerClassifier()
ticket_model.load_state_dict(torch.load(
    "synt_ticket_model_weights.pth",
    weights_only=True))
ticket_model.eval()
# ticket_synt end

app = FastAPI()


class Chunk(object):
    def __init__(self, s: str, img_url: str = None, img_pos: int = None, begin: int = None, end: int = None):
        self.s = s
        self.i = 0
        self.img = img_url
        self.img_pos = img_pos
        if (begin == None and end == None):
            self.begin = 0
            self.end = len(s)
        else:
            self.begin = begin
            self.end = end


    def if_img_for_emb_view(self):
        if self.img == None:
            return [self.s.strip()], None
        right_before = self.s[self.img_pos:].find('](') + self.img_pos + 2
        right_after = self.img_pos + len(self.img) - 1
        text = (self.s[:right_before] + "<image>" + self.s[right_after:]).strip()
        image = self.s[right_before:right_after].split(" ")[0]
        return [text], [image]


    def split_by_img(self) -> List:
        regex = r'!\[[^\]]*\]\((https?://[^\s)]+?\.(?:a?png|jpe?g|jfif|pjpeg|pjp|webp|gif|avif|bmp|tiff?|ico|cur))(?:\s+["\'][^"\']*["\'])?\)'
        matches = list(re.finditer(regex, self.s, re.IGNORECASE))
        if not matches:
            return [self]
        chunks = []
        start = 0
        pref_repl_end = 0
        link_start = matches[0].start()
        link_end = matches[0].end()
        img_url = None
        img_pos = None
        for i in range(len(matches) - 1):
            end = matches[i + 1].start()
            content = " " * (pref_repl_end - start) + self.s[pref_repl_end:end]
            img_url = self.s[link_start:link_end]
            img_pos = link_start - start
            chunks.append(Chunk(content, img_url, img_pos, self.begin + start, self.begin + end))
            img_url = None
            img_pos = None
            start = matches[i].start()
            pref_repl_end = matches[i].end()
            link_start = matches[i+1].start()
            link_end = matches[i+1].end()
        content = self.s[start:]
        img_url = self.s[link_start:link_end]
        img_pos = link_start - start
        chunks.append(Chunk(content, img_url, img_pos, self.begin + start, self.end))
        return chunks


    def split_by_const(self, max_len: int) -> List:
        end = len(self.s)
        if max_len >= end:
            return [self]
        overlap_len = max_len // 2
        res = []
        start = 0
        while start < len(self.s):
            end = min(start + max_len, len(self.s))
            img_url = None
            img_pos = None
            if self.img != None and start <= self.img_pos < end:
                img_url = self.img
                img_pos = self.img_pos - start
            res.append(Chunk(self.s[start:end], img_url, img_pos, self.begin + start, self.begin + end))
            if end == len(self.s):
                break
            start = end - overlap_len
        return res


def flatten(xss):
    return [x for xs in xss for x in xs]


def delete_all_links(text: str) -> str:
    pattern = r'!?\[[^\]\[\)\(]*\]\([^\)\(\]\[]*\)'
    return re.sub(pattern, '', text)


def to_chunks(md_content: str) -> List[Chunk]:
    result = Chunk(md_content).split_by_img()
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


async def index_one(hash, client, prev_last_id_chunk_count, url, chunks):
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
        if images != None:
            payload["img_url"] = images[0]
        id = prev_last_id + chunk.i
        points.append(PointStruct(id=id, vector=vector, payload=payload))

    client.upsert(
        collection_name=hash,
        points=points
    )


async def index(client: QdrantClient, searcher: GoogleGCS, query: str):
    emb_size = OLLAMA_EMBEDDING_MODEL_DIM
    md5_hash = hashlib.new('md5')
    md5_hash.update((query + str(time.time())).encode())
    hash = md5_hash.hexdigest()
    if client.collection_exists(collection_name=hash) == False:
        client.create_collection(
            collection_name=hash,
            vectors_config=VectorParams(
                size=emb_size,
                distance=Distance.COSINE,
                on_disk=False,
                hnsw_config=HnswConfigDiff(ef_construct=100, m=16, on_disk=False)),
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
        await asyncio.gather(*(index_one(hash, client, prev_last_ids_dict[url], url, chunks) for url, chunks in url_md_dict.items()))
        return hash, chunk_count_pred
    except Exception as e:
        print(f"Исключение во время индексации: {e}")
        return None, None


async def get_vector(texts: List, images: List) -> List[float]:
    response = await AsyncClient(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT).embed(
        model=OLLAMA_EMBEDDING_MODEL,
        input=texts
    )
    return response["embeddings"][0]


def retrieve_neighbors(client: QdrantClient, collection_name: str, results: Any) -> List:
    results = sorted(results, key=lambda x: x.id)
    neighbor_ids = set()
    prev_ci = None
    useless = set()
    for result in results:
        center_id = result.id
        neighbor_ids.update([center_id - 2, center_id - 1, center_id + 1, center_id + 2])  # на расстоянии 2
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


def combine_results(results) -> List[str]:
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
        begin = i.payload["end"]
        prev_id = i.id
    for i in range(len(scores)):
        scores[i] *= (MAGIC_COEF + len(images[i]) / chunks_cnts[i])
    return [list(t) for t in zip(*sorted(zip(scores, concat, images), reverse=True))]


async def get_relevant_documents(client: QdrantClient, collection_name: str, query: str, search_k: int):
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


async def ollama_chat_completion(messages, temperature=1.0, seed=None):
    options = {"temperature": temperature, "num_ctx": NUM_CTX}
    if seed is not None:
        options["seed"] = seed
    response = await AsyncClient(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT).chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options=options,
        # keep_alive=0
    )
    return response['message']['content']


@app.get("/publication")
async def get_body(request: Request):
    try:
        payload = await request.json()
        query = payload["query"]
        md5_hash = hashlib.new('md5')
        md5_hash.update((query).encode())
        key = "publication-" + md5_hash.hexdigest()
        cached = REDIS_CLIENT.get(key)
        if cached is None:
            hash, chunk_count = await index(CLIENT, SEARCHER, query)
            if hash is None:
                raise Exception("Индексация - FAIL")
            _, texts, images = await get_relevant_documents(CLIENT, hash, query, math.ceil(chunk_count * SEARCH_K_COEF))
            if not CLIENT.delete_collection(hash):
                raise Exception("Удаление временных индексов - FAIL")
            images = list(images[0].values())
            if len(images) > 0:
                imgs_ret = [images[0]]
                for i in range(len(images)):
                    if images[i] != imgs_ret[-1]:
                        imgs_ret.append(images[i])
            else:
                imgs_ret = None
            text = delete_all_links(delete_all_links(texts[0]))
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"{text}\nСгенерируй пост на основе текста выше, будто я сам это пишу. "
                        f"Никакой разметки. Никаких ссылок. Только живой, личный текст от первого лица."
                    )
                },
                {
                    "role": "system",
                    "content": (
                        "Ты — копирайтер, создающий посты для социальных сетей. "
                        "Ты пишешь короткий, живой и личный пост **от первого лица**. "
                        "Пост должен быть написан **без markdown-разметки**, **без ссылок**, **без изображений** и **без отстранённых комментариев**. "
                        "Не используй фразы вроде 'в этом тексте говорится' или 'в данном документе'. "
                        "Цель — вовлечь читателя, а не анализировать текст."
                    )
                },
            ]
            response = await ollama_chat_completion(messages, temperature=TEMP, seed=420)
            json_content = {"text": response, "images": imgs_ret}
            REDIS_CLIENT.set(key, json.dumps(json_content))
        else:
            json_content = json.loads(cached)
        return json_content
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/sentiment")
async def get_body(request: Request):
    try:
        payload = await request.json()
        comment = payload["comment"]
        response = sentiment_model(comment)[0]
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ticket_synt")
async def get_body(request: Request):
    try:
        payload = await request.json()
        comment = payload["comment"]
        emb = seq_model.encode([comment], show_progress_bar=False)[0]
        pred = torch.argmax(ticket_model(torch.from_numpy(emb))).item()
        return {"support_needed": bool(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ticket_llm")
async def get_body(request: Request):
    try:
        payload = await request.json()
        comment = payload["comment"]
        messages = [
            {
                "role": "system",
                "content": f"You are SMM-agent. You mark comments. If creation of ticket for support team needed, you answer:\n{STR_PASS}\nElse you answer:\n{STR_NO_ANSWER}\n."
            }, {
                "role": "user",
                "content": f"Comment:\n{comment}"
            }
        ]
        r = {"support_needed": False}
        response = await ollama_chat_completion(messages, temperature=TEMP)
        if STR_PASS in response:
            r["support_needed"] = True
        return r
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/sum")
async def get_body(request: Request):
    try:
        payload = await request.json()
        comments = payload["comments"]
        messages = [
            {
                "role": "system",
                "content": "# Вы Агент-модератор.\n## Вы суммаризируете комментарии к посту, очень кратко описываете их содержание. Надо выделить важную информацию. Запрещено считать метрики. Запрещено делать выводы. Запрещено давать рекомендации."
            }, {
                "role": "user",
                "content": f"### Комментарии:\n{comments}"
            }
        ]
        response = await ollama_chat_completion(messages, temperature=TEMP)
        return {"response": "### Краткое содержание комментариев:\n" + response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/fix")
async def get_body(request: Request):
    try:
        payload = await request.json()
        text = payload["text"]
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — языковой редактор. Твоя задача — исправить только орфографические, пунктуационные и синтаксические ошибки в тексте пользователя. "
                    "Не переписывай текст. Не меняй стиль, структуру или лексику. "
                    "Не сокращай и не добавляй предложения. Просто исправь ошибки, сохранив исходную формулировку максимально точно. "
                    "Отвечай только исправленным текстом, без комментариев и пояснений."
                )
            },
            {
                "role": "user",
                "content": text
            }
        ]
        response = await ollama_chat_completion(messages, temperature=TEMP)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ans")
async def get_body(request: Request):
    try:
        payload = await request.json()
        comment = payload["comment"]
        style = payload.get("style", "дружелюбном")
        messages = [
            {
                "role": "system",
                "content": f"Вы SMM-агент. Вы кратко отвечаете на комментарий. Если комментарий является спамом (spam), скамом (scam), оскорбительным (offensive), Вы пишете:\n{STR_NO_ANSWER}\nЕсли комментарий необходимо направить в поддержку, Вы пишете:\n{STR_PASS}\nИначе необходимо ответить пользователю в {style} стиле."
            }, {
                "role": "user",
                "content": f"Комментарий:\n{comment}"
            }
        ]
        no_answer = 0
        support_needed = 0
        r = {"no_answer": False, "support_needed": False}

        responses = await asyncio.gather(*[
            ollama_chat_completion(messages, temperature=TEMP, seed=i)
            for i in range(3)
        ])
        
        for i, response in enumerate(responses):
            if STR_PASS in response:
                support_needed += 1
            if STR_NO_ANSWER in response:
                no_answer += 1
            r[f"answer_{i}"] = response
        if no_answer != 0:
            r["no_answer"] = True
            r.pop('answer_0', None)
            r.pop('answer_1', None)
            r.pop('answer_2', None)
        if support_needed != 0:
            r["support_needed"] = True
            r.pop('answer_0', None)
            r.pop('answer_1', None)
            r.pop('answer_2', None)
        return r
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    print("Сервис работает.")
    uvicorn.run(app, host=HOST, port=PORT)
