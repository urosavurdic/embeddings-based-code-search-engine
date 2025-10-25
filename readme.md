# Code Search Engine

Implementation of an embeddings-based code search engine with fine-tuning on the CoSQA dataset.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline evaluation
python evaluation/evaluation.py

# Train the model
python fine_tuning/tuner.py --wandb --max_epochs 10 --batch_size 32

# View detailed analysis
jupyter notebook report.ipynb
```

## Project Structure

```
├── data/
│   ├── cosqa_dataset.py       # Dataset loading and preprocessing
│   ├── cosqa_module.py        # PyTorch Lightning DataModule
│   └── cosqa.ipynb            # Data exploration
├── embeddings/
│   ├── search_engine.py       # FAISS-based search engine
│   └── embeddings.ipynb       # Embeddings exploration
├── evaluation/
│   ├── evaluation.py          # Metrics (Recall@10, MRR@10, NDCG@10)
│   └── evaluator_callback.py  # Training callback
├── fine_tuning/
│   ├── code_search_model.py   # Bi-encoder with InfoNCE loss
│   └── tuner.py               # Training script
├── report.ipynb               # Complete analysis and results
├── report                     # Images from wandb
├── requirements.txt
└── README.md
```
## Approach

1. **Explored the dataset** - Understood query-code pairs and corpus structure
2. **Implemented search engine** - Used sentence-transformers + FAISS
3. **Evaluated baseline** - Pretrained model performance
4. **Fine-tuned model** - Contrastive learning with InfoNCE loss
5. **Analyzed results** - Training curves and metric improvements

## Task Components

### 1. Embeddings-based Search Engine
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: FAISS (cosine similarity)
- **Features**: Index building, search, save/load

### 2. Evaluation
- **Metrics**: Recall@10, MRR@10, NDCG@10
- **Implementation**: torchmetrics library
- **Dataset**: CoSQA from HuggingFace

### 3. Fine-tuning
- **Loss**: InfoNCE (contrastive learning)
- **Architecture**: Bi-encoder
- **Optimizer**: AdamW with warmup