import argparse
import importlib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import pytorch_lightning as pl
import wandb
from evaluation.evaluator_callback import RetrievalMetricsCallback


from code_search_model import CodeSearchModel
from data.cosqa_module import CoSQADataModule 


def _setup_parser():
    """
    Setup argument parser for code search training.
    """
    parser = argparse.ArgumentParser(add_help=False)

    # Base args
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--project_name", type=str, default="code-search")
    parser.add_argument("--experiment_name", type=str, default="exp-001")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Model args
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=10000)

    # Data args
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_path", type=str, default="data/")

    # Trainer args
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32)

    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    # Instantiate model and data
    model = CodeSearchModel(
        model_name=args.model_name,
        lr=args.lr,
        temperature=args.temperature,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )

    data = CoSQADataModule(args)

    # Load from checkpoint if provided
    if args.load_checkpoint:
        print(f"[INFO] Loading from checkpoint: {args.load_checkpoint}")
        model = CodeSearchModel.load_from_checkpoint(args.load_checkpoint)

    # Logger setup
    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.wandb:
        logger = pl.loggers.WandbLogger()
        logger.watch(model)
        logger.log_hyperparams(vars(args))

    # Callbacks
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val/loss", mode="min", patience=5
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )
    callbacks = [early_stopping, checkpoint]

    # Trainer setup
    trainer_args = {
    "max_epochs": args.max_epochs,
    "accelerator": args.accelerator,
    "devices": args.devices,
    "precision": args.precision,
    }

    retrieval_callback = RetrievalMetricsCallback(datamodule=data, k=10, use_wandb=args.wandb)

    callbacks = [early_stopping, checkpoint, retrieval_callback]

    trainer = pl.Trainer(**trainer_args, logger=logger, callbacks=callbacks)


    # Training
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)

    # Save log best model
    best_model_path = checkpoint.best_model_path
    if best_model_path:
        print(f"[INFO] Best model saved at: {best_model_path}")
        if args.wandb:
            wandb.save(best_model_path)
            print("[INFO] Uploaded best model to Weights & Biases.")


if __name__ == "__main__":
    main()