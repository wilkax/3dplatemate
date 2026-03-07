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
_ROOT_DIR = Path(__file__).parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

app = FastAPI(
    title="3DPLATEMATE.COM",
    description=(
        "Upload a photo of your build plate and receive a ready-to-slice 3MF "
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
app.include_router(generate_stl.router,     prefix="/api/v1", tags=["Generate 3MF"])
app.include_router(printers.router,         prefix="/api/v1", tags=["Printers"])
app.include_router(debug_detection.router,  prefix="/api/v1", tags=["Debug"])


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(_FRONTEND_DIR / "index.html")


@app.get("/details", include_in_schema=False)
def details():
    return FileResponse(_FRONTEND_DIR / "details.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(_ROOT_DIR / "favicon.ico")


@app.get("/details-step1.jpg", include_in_schema=False)
def details_step1():
    return FileResponse(_ROOT_DIR / "details-step1.jpg", media_type="image/jpeg")


@app.get("/details-step2.jpg", include_in_schema=False)
def details_step2():
    return FileResponse(_ROOT_DIR / "details-step2.jpg", media_type="image/jpeg")


@app.get("/details-step3.jpg", include_in_schema=False)
def details_step3():
    return FileResponse(_ROOT_DIR / "details-step3.jpg", media_type="image/jpeg")


@app.get("/details-step4.jpg", include_in_schema=False)
def details_step4():
    return FileResponse(_ROOT_DIR / "details-step4.jpg", media_type="image/jpeg")


app.mount("/", StaticFiles(directory=_FRONTEND_DIR), name="frontend")

