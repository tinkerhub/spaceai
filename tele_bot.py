import logging
from core.ai import query_result
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes, 
    MessageHandler, 
    filters
)
import os
import dotenv

dotenv.load_dotenv("ops/.env")

token = os.getenv('TELEGRAM_BOT_TOKEN')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

memory = {}

async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    chat_id = update.effective_chat.id

    history = memory.get(chat_id, [])
    response, history = query_result(text, messages=history)
    memory[chat_id] = history

    await context.bot.send_message(chat_id=chat_id, text=response)

if __name__ == '__main__':
    application = ApplicationBuilder().token(token).read_timeout(30).write_timeout(30).build()
    response_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), respond)
    application.add_handler(response_handler)
    application.run_polling()