"""
models.py — Pydantic request/response schemas for LexSearch API
"""

from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    filter_clause_type: Optional[str] = None


class SourceResult(BaseModel):
    filename: str
    page: int
    clause_type: str
    score: float
    preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResult]
    chunks_retrieved: int
    chunks_used: int
    session_id: str


class UploadResponse(BaseModel):
    filename: str
    doc_id: str
    doc_type: str
    page_count: int
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    model: str


class ClearResponse(BaseModel):
    message: str
    session_id: str
