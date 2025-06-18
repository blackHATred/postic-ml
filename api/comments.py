"""API эндпоинты для анализа комментариев."""
import torch
import asyncio
from fastapi import HTTPException

from config.settings import STR_NO_ANSWER, STR_PASS, TEMP
from services.llm_service import ollama_chat_completion
from models.requests import SentimentRequest, TicketRequest, SummaryRequest, FixTextRequest, AnswerCommentRequest
from models.responses import SentimentResponse, TicketResponse, SummaryResponse, FixTextResponse, AnswerCommentResponse


async def get_sentiment(request: SentimentRequest, sentiment_model) -> SentimentResponse:
    """Анализ тональности комментария."""
    try:
        comment = request.comment
        response = sentiment_model(comment)[0]
        return SentimentResponse(label=response['label'], score=response['score'])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def get_ticket_synt(request: TicketRequest, seq_model, ticket_model) -> TicketResponse:
    """Определение необходимости поддержки с помощью ML модели."""
    try:
        comment = request.comment
        emb = seq_model.encode([comment], show_progress_bar=False)[0]
        pred = torch.argmax(ticket_model(torch.from_numpy(emb))).item()
        return TicketResponse(support_needed=bool(pred))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def get_ticket_llm(request: TicketRequest) -> TicketResponse:
    """Определение необходимости поддержки с помощью LLM."""
    try:
        comment = request.comment
        messages = [
            {
                "role": "system",
                "content": f"You are SMM-agent. You mark comments. If creation of ticket for support team needed, you answer:\n{STR_PASS}\nElse you answer:\n{STR_NO_ANSWER}\n."
            }, {
                "role": "user",
                "content": f"Comment:\n{comment}"
            }
        ]
        response = await ollama_chat_completion(messages, temperature=TEMP)
        support_needed = STR_PASS in response
        return TicketResponse(support_needed=support_needed)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def get_summary(request: SummaryRequest) -> SummaryResponse:
    """Суммаризация комментариев к посту."""
    try:
        comments = request.comments
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
        return SummaryResponse(response="### Краткое содержание комментариев:\n" + response)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def fix_text(request: FixTextRequest) -> FixTextResponse:
    """Исправление орфографических и пунктуационных ошибок."""
    try:
        text = request.text
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
        return FixTextResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def answer_comment(request: AnswerCommentRequest) -> AnswerCommentResponse:
    """Генерация ответа на комментарий."""
    try:
        comment = request.comment
        style = request.style
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
        
        return AnswerCommentResponse(**r)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
