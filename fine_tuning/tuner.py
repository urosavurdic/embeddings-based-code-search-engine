import argparse
import logging

import pytorch_lightning as pl
import wandb

from evaluation.evaluator_callback import RetrievalMetricsCallback
from fine_tuning.code_search_model import CodeSearchModel
from data.cosqa_module import CoSQADataModule

logger = logging.getLogger(__name__)


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
    logging.basicConfig(level=logging.INFO)
    parser = _setup_parser()
    args = parser.parse_args()

    model = CodeSearchModel(
        model_name=args.model_name,
        lr=args.lr,
        temperature=args.temperature,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )

    data = CoSQADataModule(args)

    if args.load_checkpoint:
        logger.info("Loading from checkpoint: %s", args.load_checkpoint)
        model = CodeSearchModel.load_from_checkpoint(args.load_checkpoint)

    pl_logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.wandb:
        pl_logger = pl.loggers.WandbLogger(project=args.project_name, name=args.experiment_name)
        pl_logger.watch(model)
        pl_logger.log_hyperparams(vars(args))

    early_stopping = pl.callbacks.EarlyStopping(monitor="val/loss", mode="min", patience=5)
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath="training/checkpoints",
        filename="{epoch:03d}-{val_loss:.3f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )
    retrieval_callback = RetrievalMetricsCallback(datamodule=data, k=10, use_wandb=args.wandb)
    callbacks = [early_stopping, checkpoint, retrieval_callback]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=pl_logger,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)

    best_model_path = checkpoint.best_model_path
    if best_model_path:
        logger.info("Best model saved at: %s", best_model_path)
        if args.wandb:
            wandb.save(best_model_path)
            logger.info("Uploaded best model to Weights & Biases.")


if __name__ == "__main__":
    main()