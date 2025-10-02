from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from src.api.routes import router

app = FastAPI(title="Multimodal Alzheimer's Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use absolute paths and ensure static directory exists to avoid startup errors
BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = BASE_DIR / "static"
WEB_DIR = BASE_DIR / "web"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.include_router(router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")
