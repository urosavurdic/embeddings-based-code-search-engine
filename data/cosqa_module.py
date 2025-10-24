import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from .cosqa_dataset import CoSQADataset

BATCH_SIZE = 32
NUM_WORKERS = 2

class CoSQADataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the CoSQA code search dataset.

    Handles loading, splitting and DataLoader creation for training, validation, and testing.
    """

    def __init__(self, args: argparse.Namespace = None):
        """
        Initialize the DataModule. Arguments can be passed via argparse.Namespace.
        """
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get('batch_size', BATCH_SIZE)
        self.num_workers = self.args.get('num_workers', NUM_WORKERS)

        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        self.data_train: Optional[CoSQADataset] = None
        self.data_val: Optional[CoSQADataset] = None
        self.data_test: Optional[CoSQADataset] = None
        
        self.code_corpus: List[str] = []
        self.num_unique_codes: int = 0


    @classmethod
    def data_directory_path(cls) -> Path:
        """
        Returns the path to the directory where the dataset is cached.
        """
        return Path(__file__).resolve().parents[1] / 'data' / 'cosqa'
    
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Adds DataModule specific arguments to the parser.
        """
        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for DataLoaders.')
        parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='Number of workers for DataLoaders.')
        parser.add_argument('--val_split', type=float, default=0.1, help='Proportion of training data to use for validation.')

        return parser
    
    def prepare_data(self) -> None:
        """
        Download or prepare data if needed. This method is called only once.
        """
        # Data is loaded directly from Hugging Face datasets in CoSQADataset, so no action needed here.
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for different stages.
        """
        if stage == 'fit' or stage is None:
            full_dataset = CoSQADataset(split='trainval')
            val_size = int(len(full_dataset) * self.args.get('val_split', 0.1))
            train_size = len(full_dataset) - val_size
            self.data_train, self.data_val = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

            self.code_corpus = full_dataset.code_corpus
            self.num_unique_codes = len(self.code_corpus)

            print(f"[INFO] Train size: {len(self.data_train)}")
            print(f"[INFO] Val size: {len(self.data_val)}")

        if stage == 'test' or stage is None:
            self.data_test = CoSQADataset(split='test')

            print(f"[INFO] Test size: {len(self.data_test)}")
        
    def train_dataloader(self) -> DataLoader:
        """
        Returns DataLoader for training data.
        """
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.on_gpu, collate_fn=self.collate_fn)
    
    def val_dataloader(self) -> DataLoader:
        """
        Returns DataLoader for validation data.
        """
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.on_gpu, collate_fn=self.collate_fn)
    
    def test_dataloader(self) -> DataLoader:
        """
        Returns DataLoader for test data.
        """
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.on_gpu, collate_fn=self.collate_fn)
    

    @property
    def train_dataset(self):
        return self.data_train

    @property
    def val_dataset(self):
        return self.data_val

    @property
    def test_dataset(self):
        return self.data_test

    @staticmethod
    def collate_fn(batch: List[Tuple[str, str, int]]) -> Dict[str, Any]:
        """
        Custom collate function to batch data samples.

        Args:
            batch: List of tuples (query, code, index).

        Returns:
            Dictionary with batched 'queries', 'codes', and 'indices'.
        """
        queries, codes, indices = zip(*batch)
        return {
            'queries': list(queries),
            'codes': list(codes),
            'indices': torch.tensor(indices, dtype=torch.long)
        }
    
    def get_code_corpus(self) -> List[str]:
        """
        Get the unique code corpus for building search index.

        Returns:
            List of unique code snippets.
        """
        return self.code_corpus

    def get_relevance_dict(self, dataset: CoSQADataset) -> Dict[str, List[int]]:
        """
        Build relevance dictionary mapping queries to relevant code indices.

        Args:
            dataset: CoSQADataset instance.

        Returns:
            Dictionary mapping query string to list of relevant code indices.
        """
        relevance = defaultdict(list)
        for query, code, code_idx in dataset:
            relevance[query].append(code_idx)
        return dict(relevance)
    
    def configuration(self) -> Dict:
        """
        Returns the configuration of the DataModule.
        """
        return {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'num_unique_codes' : self.num_unique_codes,
            'code_corpus_size' : len(self.code_corpus),
            }
    def __repr__(self) -> str:
        """
        String representation of the data module.
        """
        basic = (
            "CoSQA Code Search DataModule\n"
            f"Batch size: {self.batch_size}\n"
            f"Num workers: {self.num_workers}\n"
        )
        
        if self.data_train is None and self.data_test is None:
            return basic
        
        data = ""
        if self.data_train is not None:
            data += f"Train size: {len(self.data_train)}\n"
        if self.data_val is not None:
            data += f"Val size: {len(self.data_val)}\n"
        if self.data_test is not None:
            data += f"Test size: {len(self.data_test)}\n"
        
        data += f"Code corpus size: {len(self.code_corpus)}\n"
        
        return basic + data

    

def load_and_print_info(data_module_class: type) -> None:
    parser = argparse.ArgumentParser()
    data_module_class.add_arguments(parser)
    args = parser.parse_args()

    data_module = data_module_class(args)
    data_module.prepare_data()
    data_module.setup()

    print(f"Number of training samples: {len(data_module.data_train)}")
    print(f"Number of validation samples: {len(data_module.data_val)}")
    print(f"Number of test samples: {len(data_module.data_test)}")
    print(f"Number of unique code snippets in corpus: {data_module.num_unique_codes}")


if __name__ == "__main__":
    load_and_print_info(CoSQADataModule)