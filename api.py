import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from data.cosqa_dataset import CoSQADataset
from embeddings.search_engine import EmbeddingSearchEngine

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH")
INDEX_PATH = os.environ.get("INDEX_PATH")
DOCS_PATH = os.environ.get("DOCS_PATH")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "test")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)

    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        from fine_tuning.code_search_model import CodeSearchModel
        model = CodeSearchModel.load(CHECKPOINT_PATH)
        engine = EmbeddingSearchEngine(encoder=model.get_encoder())
        logger.info("Loaded fine-tuned model from checkpoint: %s", CHECKPOINT_PATH)
    else:
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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo():
    return HTMLResponse("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Code Search Engine</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; min-height: 100vh; }
  header { background: #161b22; border-bottom: 1px solid #30363d; padding: 18px 32px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.2rem; font-weight: 600; color: #f0f6fc; }
  header span { font-size: 0.75rem; background: #1f6feb; color: #fff; padding: 2px 8px; border-radius: 12px; }
  main { max-width: 860px; margin: 48px auto; padding: 0 24px; }
  .search-box { display: flex; gap: 10px; margin-bottom: 12px; }
  input[type=text] { flex: 1; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; font-size: 1rem; color: #f0f6fc; outline: none; transition: border-color .2s; }
  input[type=text]:focus { border-color: #1f6feb; }
  .controls { display: flex; align-items: center; gap: 16px; margin-bottom: 32px; }
  label { font-size: 0.85rem; color: #8b949e; }
  select { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 6px 10px; color: #c9d1d9; font-size: 0.85rem; }
  button { background: #1f6feb; color: #fff; border: none; border-radius: 8px; padding: 12px 24px; font-size: 1rem; cursor: pointer; font-weight: 500; transition: background .2s; white-space: nowrap; }
  button:hover { background: #388bfd; }
  button:disabled { background: #30363d; cursor: not-allowed; }
  .hint { font-size: 0.82rem; color: #6e7681; margin-bottom: 32px; }
  .hint code { background: #161b22; padding: 1px 6px; border-radius: 4px; font-family: monospace; color: #79c0ff; }
  .result { background: #161b22; border: 1px solid #30363d; border-radius: 10px; margin-bottom: 16px; overflow: hidden; }
  .result-header { display: flex; justify-content: space-between; align-items: center; padding: 10px 16px; background: #1c2128; border-bottom: 1px solid #30363d; }
  .rank { font-size: 0.8rem; font-weight: 600; color: #8b949e; }
  .score { font-size: 0.8rem; font-weight: 600; color: #3fb950; }
  pre { padding: 16px; overflow-x: auto; font-family: "Fira Code", "Cascadia Code", monospace; font-size: 0.85rem; line-height: 1.6; color: #e6edf3; }
  #status { color: #8b949e; font-size: 0.9rem; margin-bottom: 16px; min-height: 22px; }
  .error { color: #f85149; }
</style>
</head>
<body>
<header>
  <svg width="20" height="20" viewBox="0 0 16 16" fill="#58a6ff"><path d="M2 2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75h-2.5a.75.75 0 0 1 0-1.5h1.75v-2h-8a1 1 0 0 0-.714 1.7.75.75 0 1 1-1.072 1.05A2.495 2.495 0 0 1 2 11.5Zm10.5-1h-8a1 1 0 0 0-1 1v6.708A2.486 2.486 0 0 1 4.5 9h8Z"/></svg>
  <h1>Code Search Engine</h1>
  <span>fine-tuned · Recall@10 99.7%</span>
</header>
<main>
  <div class="search-box">
    <input type="text" id="query" placeholder="e.g. sort a list in reverse order" autofocus>
    <button id="btn" onclick="search()">Search</button>
  </div>
  <div class="controls">
    <label>Results: <select id="topk"><option>3</option><option selected>5</option><option>10</option></select></label>
  </div>
  <p class="hint">Try: <code>read a file line by line</code> &nbsp;·&nbsp; <code>convert string to lowercase</code> &nbsp;·&nbsp; <code>check if key exists in dict</code></p>
  <div id="status"></div>
  <div id="results"></div>
</main>
<script>
  document.getElementById("query").addEventListener("keydown", e => { if (e.key === "Enter") search(); });

  async function search() {
    const q = document.getElementById("query").value.trim();
    if (!q) return;
    const topk = document.getElementById("topk").value;
    const btn = document.getElementById("btn");
    const status = document.getElementById("status");
    const results = document.getElementById("results");

    btn.disabled = true;
    btn.textContent = "Searching…";
    status.textContent = "";
    results.innerHTML = "";

    try {
      const res = await fetch(`/search?q=${encodeURIComponent(q)}&top_k=${topk}`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      status.textContent = `${data.results.length} results for "${data.query}"`;
      results.innerHTML = data.results.map(r => `
        <div class="result">
          <div class="result-header">
            <span class="rank">#${r.rank}</span>
            <span class="score">score ${r.score.toFixed(4)}</span>
          </div>
          <pre>${escHtml(r.code)}</pre>
        </div>`).join("");
    } catch (e) {
      status.innerHTML = `<span class="error">Error: ${e.message}</span>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Search";
    }
  }

  function escHtml(s) {
    return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  }
</script>
</body>
</html>""")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
