import dotenv
import json

dotenv.load_dotenv("ops/.env")


with open("data/role.json", "r") as f:
    role = json.load(f)


def is_admin(user_id):
    if user_id in role["admin"]:
        return True
    return False

