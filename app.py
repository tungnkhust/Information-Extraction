from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import HTMLResponse
import uvicorn
import time
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from api.router.router import model_router

app = FastAPI()
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)

app.include_router(model_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")