import uvicorn
from fastapi import FastAPI
from core.routes import router
from core.telegram import check_telegram_webhook
app = FastAPI()

app.include_router(router.router)

if __name__ == "__main__":
    check_telegram_webhook()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)