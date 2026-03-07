from pydantic import BaseModel
from typing import Optional, List


class SearchResult(BaseModel):
    content: str
    source: str
    score: float
    location: Optional[str] = None
    tome: Optional[str] = None
    part: Optional[str] = None
    chapter: Optional[str] = None
    epilogue: Optional[str] = None
    source_version: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]


class AskResponse(BaseModel):
    answer: str
    sources: List[SearchResult]
