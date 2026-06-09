import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from data.cosqa_dataset import CoSQADataset
from embeddings.search_engine import EmbeddingSearchEngine

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
INDEX_PATH = os.environ.get("INDEX_PATH")
DOCS_PATH = os.environ.get("DOCS_PATH")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "test")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    engine = EmbeddingSearchEngine(model_name=MODEL_NAME)

    if INDEX_PATH and os.path.exists(INDEX_PATH):
        docs_path = DOCS_PATH or INDEX_PATH.replace(".faiss", "_docs.npy")
        engine.load_index(INDEX_PATH, docs_path)
        logger.info("Loaded prebuilt index from %s", INDEX_PATH)
        state["index_source"] = INDEX_PATH
    else:
        logger.info("Building index from CoSQA %s split...", DATASET_SPLIT)
        dataset = CoSQADataset(split=DATASET_SPLIT)
        engine.build_index(dataset.code_corpus)
        state["index_source"] = f"CoSQA:{DATASET_SPLIT}"

    state["engine"] = engine
    yield
    state.clear()


app = FastAPI(
    title="Code Search Engine",
    description="Embeddings-based semantic search over the CoSQA code corpus.",
    version="0.1.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")


class SearchHit(BaseModel):
    rank: int
    score: float
    code: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchHit]


class IndexRequest(BaseModel):
    documents: List[str] = Field(..., min_length=1, description="Documents to index")


class IndexResponse(BaseModel):
    status: str
    num_documents: int
    source: str


class HealthResponse(BaseModel):
    status: str
    model: str
    documents: int
    index_source: str


def _get_engine() -> EmbeddingSearchEngine:
    engine = state.get("engine")
    if engine is None or engine.index is None:
        raise HTTPException(status_code=503, detail="Search engine not ready")
    return engine


def _run_search(query: str, top_k: int) -> SearchResponse:
    engine = _get_engine()
    raw = engine.search(query, top_k=top_k)
    results = [SearchHit(rank=i, score=score, code=code) for i, (code, score) in enumerate(raw, 1)]
    return SearchResponse(query=query, results=results)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    engine = _get_engine()
    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        documents=engine.index.ntotal,
        index_source=state.get("index_source", "unknown"),
    )


@app.get("/search", response_model=SearchResponse)
def search_get(
    q: str = Query(..., min_length=1, description="Natural language query"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results to return"),
) -> SearchResponse:
    return _run_search(q, top_k)


@app.post("/search", response_model=SearchResponse)
def search_post(request: SearchRequest) -> SearchResponse:
    return _run_search(request.query, request.top_k)


@app.post("/index", response_model=IndexResponse)
def reindex(request: IndexRequest) -> IndexResponse:
    engine = EmbeddingSearchEngine(model_name=MODEL_NAME)
    engine.build_index(request.documents)
    state["engine"] = engine
    state["index_source"] = "user-supplied"
    return IndexResponse(status="ok", num_documents=engine.index.ntotal, source="user-supplied")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
