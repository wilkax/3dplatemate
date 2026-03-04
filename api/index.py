# Vercel Python serverless entry point.
# Vercel looks for a variable named `app` (ASGI) or `handler` (WSGI) in this file.
# We simply re-export the FastAPI application from the main module.
from app.main import app  # noqa: F401  — re-exported for Vercel

__all__ = ["app"]

