import uvicorn
from fastapi import FastAPI
from core.routes import router
app = FastAPI()

from dotenv import load_dotenv

load_dotenv("ops/.env")


app.include_router(router.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)