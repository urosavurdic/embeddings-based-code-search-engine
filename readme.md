# Code search engine
Task is to implement and evaluate a simple embeddings-based code search engine:
1. Embeddings-based search engine
2. Evaluation
3. Fine-tuning

My approach was to understand data and models and experiment. That can be seen in `.ipynb` files.
The structure of this project is the following:
 - `data/` - folder that contains all data related tasks:
    - `cosqa_dataset.py` - defines what should be expected from the dataset and chooses relevant coloumns
    - `cosqa_module.py` - handles data loading into train, test, val as well as splits and batches and number of workers
 - `embeddings/` - defines model
    - `search_engine.py` - contains Bi-encoder implementation using `sequence_transformer`
 - `evaluation/` - contains evaluation related tasks
    - `evaluation.py` - scribt containing 
