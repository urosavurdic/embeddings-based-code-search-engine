import logging
from collections import defaultdict
from typing import Dict, List

import torch
from tqdm import tqdm
from torchmetrics.retrieval import RetrievalMRR, RetrievalRecall, RetrievalNormalizedDCG

logger = logging.getLogger(__name__)


def build_relevance_dict(dataset) -> Dict[str, List[int]]:
    relevance = defaultdict(list)
    for i in range(len(dataset)):
        query, code, code_idx = dataset[i]
        relevance[query].append(code_idx)
    return dict(relevance)


class SearchEngineEvaluator:

    def __init__(self, search_engine, k: int = 10):
        self.search_engine = search_engine
        self.k = k
        self.recall_metric = RetrievalRecall(top_k=k)
        self.mrr_metric = RetrievalMRR(top_k=k)
        self.ndcg_metric = RetrievalNormalizedDCG(top_k=k)

    def evaluate(self, dataset, verbose: bool = True) -> Dict[str, float]:
        relevance_dict = build_relevance_dict(dataset)
        unique_queries = list(relevance_dict.keys())

        self.recall_metric.reset()
        self.mrr_metric.reset()
        self.ndcg_metric.reset()

        iterator = tqdm(unique_queries, desc="Evaluating") if verbose else unique_queries

        for query_idx, query in enumerate(iterator):
            relevant_indices = set(relevance_dict[query])
            results = self.search_engine.search(query, top_k=self.k)

            retrieved_indices = []
            scores = []
            for doc_text, score in results:
                try:
                    idx = self.search_engine.doc_index(doc_text)
                    retrieved_indices.append(idx)
                    scores.append(score)
                except KeyError:
                    continue

            if retrieved_indices:
                preds = torch.tensor(scores, dtype=torch.float32)
                target = torch.tensor(
                    [1.0 if idx in relevant_indices else 0.0 for idx in retrieved_indices],
                    dtype=torch.float32,
                )
                indexes = torch.tensor([query_idx] * len(retrieved_indices), dtype=torch.long)

                self.recall_metric.update(preds, target, indexes=indexes)
                self.mrr_metric.update(preds, target, indexes=indexes)
                self.ndcg_metric.update(preds, target, indexes=indexes)

        recall = self.recall_metric.compute().item()
        mrr = self.mrr_metric.compute().item()
        ndcg = self.ndcg_metric.compute().item()

        metrics = {
            f'Recall@{self.k}': recall,
            f'MRR@{self.k}': mrr,
            f'NDCG@{self.k}': ndcg,
            'num_queries': len(unique_queries),
        }

        if verbose:
            logger.info("Evaluation Results (K=%d)", self.k)
            logger.info("Recall@%d: %.4f", self.k, recall)
            logger.info("MRR@%d: %.4f", self.k, mrr)
            logger.info("NDCG@%d: %.4f", self.k, ndcg)
            logger.info("Unique queries: %d", len(unique_queries))

        return metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from embeddings.search_engine import EmbeddingSearchEngine
    from data.cosqa_dataset import CoSQADataset

    test_dataset = CoSQADataset(split='test')
    search_engine = EmbeddingSearchEngine()
    search_engine.build_index(test_dataset.code_corpus)

    evaluator = SearchEngineEvaluator(search_engine, k=10)
    evaluator.evaluate(test_dataset, verbose=True)