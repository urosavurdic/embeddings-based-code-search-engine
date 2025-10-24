import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
from torchmetrics.retrieval import RetrievalMRR, RetrievalRecall, RetrievalNormalizedDCG
import torch

class SearchEngineEvaluator:
    """
    Evaluates a search engine on the CoSQA dataset using library metrics.
    """
    
    def __init__(self, search_engine, k: int = 10):
        self.search_engine = search_engine
        self.k = k
        self.recall_metric = RetrievalRecall(top_k=k)
        self.mrr_metric = RetrievalMRR(top_k=k)
        self.ndcg_metric = RetrievalNormalizedDCG(top_k=k)
    
    def build_relevance_dict(self, dataset) -> Dict[str, List[int]]:
        """
        Build a dictionary mapping queries to relevant code indices.
        
        Args:
            dataset: CoSQADataset instance.
            
        Returns:
            Dictionary: {query_string: [relevant_code_idx1, relevant_code_idx2, ...]}
        """
        relevance = defaultdict(list)
        
        for i in range(len(dataset)):
            query, code, code_idx = dataset[i]
            relevance[query].append(code_idx)
        
        return dict(relevance)
    
    def evaluate(self, dataset, verbose: bool = True) -> Dict[str, float]:
        """
        Args:
            dataset: CoSQADataset instance.
            verbose: Whether to show progress bar.
            
        Returns:
            Dictionary containing average metrics.
        """
        relevance_dict = self.build_relevance_dict(dataset)
        unique_queries = list(relevance_dict.keys())
        
        # Reset metrics
        self.recall_metric.reset()
        self.mrr_metric.reset()
        self.ndcg_metric.reset()
        
        iterator = tqdm(unique_queries, desc="Evaluating") if verbose else unique_queries
        
        for query_idx, query in enumerate(iterator):
            relevant_indices = relevance_dict[query]
            
            # Perform search
            results = self.search_engine.search(query, top_k=self.k)
            
            # Build binary relevance labels and scores
            retrieved_indices = []
            scores = []
            
            for doc_text, score in results:
                try:
                    idx = self.search_engine.documents.index(doc_text)
                    retrieved_indices.append(idx)
                    scores.append(score)
                except ValueError:
                    continue
            
            if len(retrieved_indices) > 0:
                preds = torch.tensor(scores, dtype=torch.float32)
                target = torch.tensor([1.0 if idx in relevant_indices else 0.0 for idx in retrieved_indices], dtype=torch.float32)
                indexes = torch.tensor([query_idx] * len(retrieved_indices), dtype=torch.long)
                
                # Update metrics
                self.recall_metric.update(preds, target, indexes=indexes)
                self.mrr_metric.update(preds, target, indexes=indexes)
                self.ndcg_metric.update(preds, target, indexes=indexes)
        
        # Compute final metrics
        recall = self.recall_metric.compute().item()
        mrr = self.mrr_metric.compute().item()
        ndcg = self.ndcg_metric.compute().item()
        
        results = {
            f'Recall@{self.k}': recall,
            f'MRR@{self.k}': mrr,
            f'NDCG@{self.k}': ndcg,
            'num_queries': len(unique_queries)
        }
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Evaluation Results (K={self.k})")
            print(f"{'='*50}")
            print(f"Recall@{self.k}: {recall:.4f}")
            print(f"MRR@{self.k}: {mrr:.4f}")
            print(f"NDCG@{self.k}: {ndcg:.4f}")
            print(f"Number of unique queries: {len(unique_queries)}")
            print(f"{'='*50}\n")
        
        return results

if __name__ == "__main__":
    from embeddings.search_engine import EmbeddingSearchEngine
    from data.cosqa_dataset import CoSQADataset
    
    print("="*80)
    print("EMBEDDING-BASED CODE SEARCH ENGINE EVALUATION")
    print("="*80)
    
    # Load test dataset
    print("\nLoading CoSQA test dataset...")
    test_dataset = CoSQADataset(split='test')
    
    # Initialize search engine
    print("Initializing search engine...")
    search_engine = EmbeddingSearchEngine()
    
    # Build index on code corpus
    print("Building search index...")
    code_corpus = test_dataset.code_corpus
    search_engine.build_index(code_corpus)
    
    # Evaluate
    print("\nEvaluating search engine...")
    evaluator = SearchEngineEvaluator(search_engine, k=10)
    results = evaluator.evaluate(test_dataset, verbose=True)