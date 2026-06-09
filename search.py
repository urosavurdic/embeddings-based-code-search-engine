import argparse
import logging

from embeddings.search_engine import EmbeddingSearchEngine
from data.cosqa_dataset import CoSQADataset


def main():
    parser = argparse.ArgumentParser(description="Search code snippets using natural language queries.")
    parser.add_argument("query", nargs="?", help="Natural language query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model name")
    parser.add_argument("--index", type=str, default=None, help="Path to pre-built FAISS index")
    parser.add_argument("--dataset", type=str, default="test", choices=["trainval", "test"], help="Dataset split to index")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    engine = EmbeddingSearchEngine(model_name=args.model)

    if args.index:
        engine.load_index(args.index, args.index.replace(".faiss", "_docs.npy"))
    else:
        dataset = CoSQADataset(split=args.dataset)
        engine.build_index(dataset.code_corpus)

    if args.query:
        _run_query(engine, args.query, args.top_k)
    else:
        _interactive(engine, args.top_k)


def _run_query(engine, query, top_k):
    results = engine.search(query, top_k=top_k)
    for rank, (code, score) in enumerate(results, 1):
        print(f"\n--- Result {rank} (score: {score:.4f}) ---")
        print(code)


def _interactive(engine, top_k):
    print("Interactive code search (type 'quit' to exit)\n")
    while True:
        query = input("Query: ").strip()
        if not query or query.lower() == "quit":
            break
        _run_query(engine, query, top_k)
        print()


if __name__ == "__main__":
    main()
