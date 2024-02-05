from fastapi import FastAPI
from api.router import tagging_router as router

app = FastAPI()
app.include_router(router)