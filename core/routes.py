from fastapi import FastAPI, HTTPException, Request, APIRouter
from core.telegram import (
    send_telegram_message,  
    get_bot_user_id
)
from core.ai import ChatBot
from core.models import Message

bot = ChatBot()

router = APIRouter()


@router.post("/telegram")
async def process_telegram_message(request: Request):
    try:
        data = await request.json()
        if "message" not in data:
            raise HTTPException(status_code=400, detail="Invalid request data")

        message = data["message"]
        if "text" not in message or "from" not in message:
            raise HTTPException(status_code=400, detail="Invalid message format")

        chat_id = message["chat"]["id"]
        user_id = message["from"]["id"]
        text = message["text"]

        response = bot.get_response(chat_id, text)

        BOT_USER_ID = get_bot_user_id()

        if user_id == BOT_USER_ID:
            return {"message": "Ignoring bot's own message"}
        
        send_telegram_message(chat_id, response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"message": "Message processed successfully"}

@router.post("/web")
async def process_web_message(message: Message):
    chat_id = message.chat_id
    text = message.text

    response = bot.get_response(chat_id, text)

    return {"message": response}
    