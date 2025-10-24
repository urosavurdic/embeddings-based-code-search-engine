import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import Dict, Any
import wandb


class CodeSearchModel(pl.LightningModule):
    """
    Bi-encoder model for code search using contrastive (InfoNCE) loss.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        lr: float = 2e-5,
        temperature: float = 0.07,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 10000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = SentenceTransformer(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = True
        self.encoder.train()

        # Hyperparameters
        self.lr = lr
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def encode(self, texts: list) -> torch.Tensor:
        batch = self.encoder.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        ).to(self.device)

        model = self.encoder._first_module().auto_model
        outputs = model(**batch)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(embeddings, p=2, dim=1)

    def contrastive_loss(self, queries: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        sim = torch.matmul(queries, codes.T) / self.temperature
        labels = torch.arange(sim.size(0), device=self.device)
        return F.cross_entropy(sim, labels)

    def forward(self, queries: list, codes: list) -> Dict[str, torch.Tensor]:
        q_emb = self.encode(queries)
        c_emb = self.encode(codes)
        loss = self.contrastive_loss(q_emb, c_emb)
        return {"loss": loss, "q_emb": q_emb, "c_emb": c_emb}

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        out = self(batch["queries"], batch["codes"])
        loss = out["loss"]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if wandb.run is not None:
            wandb.log({"train/loss": loss.item(), "step": self.global_step})
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        out = self(batch["queries"], batch["codes"])
        loss = out["loss"]
        self.log("val/loss", loss, prog_bar=True)
        if wandb.run is not None:
            wandb.log({"val/loss": loss.item(), "step": self.global_step})
        return {"val_loss": loss}

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        out = self(batch["queries"], batch["codes"])
        loss = out["loss"]
        self.log("test/loss", loss)
        return {"test_loss": loss}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            return max(0.0, (self.max_steps - step) / max(1, self.max_steps - self.warmup_steps))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def get_encoder(self):
        return self.encoder

    def save_encoder(self, path: str):
        self.encoder.save(path)
        print(f"[INFO] Saved encoder to {path}")
