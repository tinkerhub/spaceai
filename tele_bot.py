import logging
from core.ai import chat
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes, 
    MessageHandler,
    CommandHandler, 
    filters,
)
from core.db import set_redis, get_redis_value
from core.ingest import get_pdf_doc, update_vector_db
from utils.auth import is_admin
import os
import dotenv

dotenv.load_dotenv("ops/.env")

token = os.getenv('TELEGRAM_BOT_TOKEN')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await context.bot.send_message(chat_id=chat_id, text="Hey, I'm S.P.A.C.E AI! Front desk virtual assistant for Tinkerspace. You can ask me about Tinkerhub and Tinkerspace. Go to tinkerhub.org to know more about Tinkerhub :)")

async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    chat_id = update.effective_chat.id
    print(chat_id)
    history = get_redis_value(chat_id)
    if not history:
        history = []
        set_redis(chat_id, history, expire=1800)
    response, history = chat(text, messages=history)
    set_redis(chat_id, history)
    
    await context.bot.send_message(chat_id=chat_id, text=response)

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if is_admin(chat_id):
        document = update.message.document
        file_id = document.file_id
        file = await context.bot.get_file(file_id)
        await file.download_to_drive("data/temp.pdf")
        doc = get_pdf_doc("temp.pdf")
        update_vector_db(doc)
        os.remove("data/temp.pdf")
        await context.bot.send_message(chat_id=chat_id, text="File received and vectordb updated!")
    else:
        await context.bot.send_message(chat_id=chat_id, text="We doont do that here!")

if __name__ == '__main__':
    application = ApplicationBuilder().token(token).read_timeout(30).write_timeout(30).build()
    start_handler = CommandHandler('start', start)
    response_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), respond)
    document_handler = MessageHandler(filters.Document.PDF, handle_document)
    application.add_handler(response_handler)
    application.add_handler(document_handler)
    application.add_handler(start_handler)
    application.run_polling()