"""FastAPI entry point. Run with: uvicorn app.api.main:app --reload --port 8000"""
from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="Multimodal AI Research & Teaching Assistant",
    version="0.1.0",
    description="Upload PDFs, ask grounded questions, explain figures.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Endpoints are wired up in Notebook 05.
# from app.api import routes  # noqa
