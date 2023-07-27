import redis

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# Expiration time for each chat history entry in seconds (30 minutes in this example)
CACHE_EXPIRATION_SECONDS = 1800

def get_conversation_history(chat_id):
    key = f"conversation:{chat_id}"
    conversation_history = redis_client.get(key)
    return conversation_history.decode() if conversation_history else None

def update_conversation_history(
        chat_id, history, expiration=CACHE_EXPIRATION_SECONDS
):
    key = f"conversation:{chat_id}"
    if redis_client.exists(key):
        redis_client.set(key, history)
    else:
        redis_client.setex(key, history, expiration)