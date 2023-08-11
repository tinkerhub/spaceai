import redis
import json

def get_redis():
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
    return redis_client

def set_redis(key, value, expire=1800):
    redis_client = get_redis()
    value = json.dumps(value)
    current_expiration = redis_client.ttl(key)
    redis_client.set(key, value)
    if current_expiration >= 0:
        redis_client.expire(key, current_expiration)
    else:
        redis_client.expire(key, expire)

def get_redis_value(key):
    redis_client = get_redis()
    value = redis_client.get(key)
    if value is None:
        return None
    value = json.loads(value)
    print(redis_client.ttl(key))
    print(value)
    return value

