import logging

import pytorch_lightning as pl

from embeddings.search_engine import EmbeddingSearchEngine
from evaluation.evaluation import SearchEngineEvaluator

logger = logging.getLogger(__name__)


class RetrievalMetricsCallback(pl.Callback):

    def __init__(self, datamodule, k=10, use_wandb=True, eval_every_n_epochs=1):
        super().__init__()
        self.datamodule = datamodule
        self.k = k
        self.use_wandb = use_wandb
        self.eval_every_n_epochs = eval_every_n_epochs

    def _evaluate(self, trainer, pl_module, split="val"):
        if split == "val" and (trainer.current_epoch + 1) % self.eval_every_n_epochs != 0:
            return

        if split == "val":
            dataset = self.datamodule.val_dataset
        elif split == "test":
            dataset = self.datamodule.test_dataset
        else:
            raise ValueError("split must be 'val' or 'test'")

        base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
        search_engine = EmbeddingSearchEngine(encoder=pl_module.get_encoder())
        search_engine.build_index(base_dataset.code_corpus)

        evaluator = SearchEngineEvaluator(search_engine, k=self.k)
        metrics = evaluator.evaluate(dataset, verbose=False)

        for key, value in metrics.items():
            trainer.logger.log_metrics({f"{split}/{key}": value}, step=trainer.global_step)

        logger.info("%s metrics: %s", split.upper(), metrics)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._evaluate(trainer, pl_module, split="val")

    def on_test_epoch_end(self, trainer, pl_module):
        self._evaluate(trainer, pl_module, split="test")