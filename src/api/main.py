from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path

from src.api.routes import router

app = FastAPI(title="Multimodal Alzheimer's Detection API")

# Global exception handler to avoid 500s and surface errors to frontend
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=200, content={"status": "error", "result": {"error": str(exc)}})

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

# Debug route to list available endpoints
@app.get("/debug/routes")
async def list_routes():
    return {
        "routes": [
            {"path": r.path, "methods": list(r.methods or [])}
            for r in app.router.routes
            if hasattr(r, "path")
        ]
    }

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")
