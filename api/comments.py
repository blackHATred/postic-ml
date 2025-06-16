"""API эндпоинты для анализа комментариев."""
import torch
import asyncio
from fastapi import HTTPException, Request

from config.settings import STR_NO_ANSWER, STR_PASS, TEMP
from services.llm_service import ollama_chat_completion


async def get_sentiment(request: Request, sentiment_model):
    """Анализ тональности комментария."""
    try:
        payload = await request.json()
        comment = payload["comment"]
        response = sentiment_model(comment)[0]
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def get_ticket_synt(request: Request, seq_model, ticket_model):
    """Определение необходимости поддержки с помощью ML модели."""
    try:
        payload = await request.json()
        comment = payload["comment"]
        emb = seq_model.encode([comment], show_progress_bar=False)[0]
        pred = torch.argmax(ticket_model(torch.from_numpy(emb))).item()
        return {"support_needed": bool(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def get_ticket_llm(request: Request):
    """Определение необходимости поддержки с помощью LLM."""
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


async def get_summary(request: Request):
    """Суммаризация комментариев к посту."""
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


async def fix_text(request: Request):
    """Исправление орфографических и пунктуационных ошибок."""
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


async def answer_comment(request: Request):
    """Генерация ответа на комментарий."""
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
