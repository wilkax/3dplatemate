"""
3D Print Plate Cleaner — FastAPI backend entry point.

Start the server with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import debug_detection, generate_stl, prepare, printers

_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

app = FastAPI(
    title="3D Print Plate Cleaner",
    description=(
        "Upload a photo of your build plate and receive a ready-to-slice STL "
        "with cleaner objects precisely positioned over every dirty spot."
    ),
    version="0.1.0",
)

# Allow all origins during development; tighten for production as needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prepare.router,          prefix="/api/v1", tags=["Prepare"])
app.include_router(generate_stl.router,     prefix="/api/v1", tags=["Generate STL"])
app.include_router(printers.router,         prefix="/api/v1", tags=["Printers"])
app.include_router(debug_detection.router,  prefix="/api/v1", tags=["Debug"])


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(_FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=_FRONTEND_DIR), name="frontend")

