import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import pandas as pd



class CoSQADataset(Dataset):
    """
    PyTorch Dataset for the CoSQA dataset.
    """
    def __init__(self, split: str = 'trainval'):
        """
        Initialize the dataset.

        Args:
            split: Dataset split to load ('trainval' or 'test').
        """
        if split == 'trainval':
            self.data = pd.read_json("hf://datasets/gonglinyuan/CoSQA/cosqa-train.json")
        elif split == 'test':
            self.data = pd.read_json("hf://datasets/gonglinyuan/CoSQA/cosqa-dev.json")
        else:
            raise ValueError("Invalid split name. Use 'trainval' or 'test'.")
        
        print(f"[INFO] Loading CoSQA {split} split...")
        
        self.queries = []
        self.codes = []
        self.indices = []
        
        # Build unique code corpus and query-code mappings
        code_to_idx = {}
        self.code_corpus = []
        
        for _, item in self.data.iterrows():
            query = item['doc']
            code = item['code']
            
            # Add to unique code corpus
            if code not in code_to_idx:
                code_to_idx[code] = len(self.code_corpus)
                self.code_corpus.append(code)
            
            code_idx = code_to_idx[code]
            
            self.queries.append(query)
            self.codes.append(code)
            self.indices.append(code_idx)
        
        print(f"[INFO] Loaded {len(self)} query-code pairs")
        print(f"[INFO] Unique code snippets: {len(self.code_corpus)}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.queries[idx], self.codes[idx], self.indices[idx]
