"""
Build and save FAISS index from the fine-tuned model checkpoint.

Usage:
    python build_index.py
    python build_index.py --checkpoint training/checkpoints/epoch=009-val_loss=0.000.ckpt
    python build_index.py --checkpoint training/checkpoints/epoch=009-val_loss=0.000.ckpt --split test
"""
import argparse
import logging

from data.cosqa_dataset import CoSQADataset
from embeddings.search_engine import EmbeddingSearchEngine
from fine_tuning.code_search_model import CodeSearchModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="training/checkpoints/epoch=009-val_loss=0.000.ckpt")
    parser.add_argument("--split", type=str, default="test", choices=["test", "trainval"])
    parser.add_argument("--index_path", type=str, default="index.faiss")
    parser.add_argument("--docs_path", type=str, default="documents.npy")
    args = parser.parse_args()

    logger.info("Loading fine-tuned model from %s", args.checkpoint)
    model = CodeSearchModel.load(args.checkpoint)

    logger.info("Loading CoSQA %s corpus...", args.split)
    dataset = CoSQADataset(split=args.split)

    engine = EmbeddingSearchEngine(encoder=model.get_encoder())
    engine.build_index(dataset.code_corpus)
    engine.save_index(args.index_path, args.docs_path)
    logger.info("Done. Index: %s  Docs: %s", args.index_path, args.docs_path)


if __name__ == "__main__":
    main()
