import requests
from fastapi import HTTPException
import os

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

def send_telegram_message(chat_id: int, text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text
    }
    response = requests.post(url, json=data)
    if not response.ok:
        raise HTTPException(status_code=500, detail="Failed to send the message to Telegram")

def set_telegram_webhook():
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook"
    data = {
        "url": "https://your-fastapi-webapp.com/telegram"  # Replace with your public webhook URL
    }
    response = requests.post(url, json=data)
    if not response.ok:
        print("Failed to set the Telegram webhook.")
        print(response.text)
    else:
        print("Telegram webhook has been set successfully.")

def check_telegram_webhook():
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getWebhookInfo"
    response = requests.get(url)
    if not response.ok:
        print("Failed to get webhook info from Telegram.")
        print(response.text)
    else:
        webhook_info = response.json()
        if webhook_info["ok"] and not webhook_info["result"]["url"]:
            set_telegram_webhook()
        else:
            print("Webhook is already set.")

def get_bot_user_id():
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
    response = requests.get(url)
    if not response.ok:
        print("Failed to get bot information.")
        print(response.text)
        return None
    else:
        bot_info = response.json()
        bot_user_id = bot_info["result"]["id"]
        return bot_user_id