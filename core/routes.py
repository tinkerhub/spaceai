from fastapi import APIRouter
from core.ai import query_result
from core.models import Message


router = APIRouter()

history = {}

@router.post("/web")
async def process_web_message(message: Message):
    chat_id = message.chat_id
    text = message.text
    history = history.get(chat_id, [])
    response, new_history = query_result(text, history)
    history[chat_id] = new_history
    return {"message": response}
    