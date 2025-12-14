# BCSD Model - Binary Code Similarity Detection Using GNN-Enriched BERT

A complete Binary Code Similarity Detection (BCSD) pipeline that integrates Control Flow Graph (CFG) structure via Graph Neural Networks (GNNs) with sequence-based BERT embeddings using a novel KV-Prefix attention mechanism.

**Research Hypothesis**: Joint modeling of graph structure (via GNN) and sequence semantics (via BERT) achieves higher similarity detection accuracy compared to sequence-only or graph-only baselines.

## Overview

This repository implements a complete end-to-end pipeline for binary code similarity detection:

1. **Preprocessing (angr + Custom Tokenization)** - Extracts CFGs from binaries and tokenizes assembly instructions with domain-specific vocabulary
2. **GNN Encoder** - Processes variable-size CFGs into fixed-size graph summaries using Graph Attention Networks (GAT)
3. **Custom KV-Prefix Attention** - Injects graph-derived prefixes into BERT's key-value attention mechanism (the core innovation)
4. **BERT Integration** - Deep prefix injection across all 12 BERT layers for joint graph-sequence modeling
5. **Siamese Training** - Joint MLM (Masked Language Modeling) + Contrastive Learning with configurable loss weights
6. **Inference Pipeline** - Vectorizes binaries and performs mathematical similarity search

## Project Structure

```
BCSD-Model-using-GNN-to-enrichen-V-K-/
├── preprocessing/              # US1: angr CFG extraction + assembly tokenization
│   ├── extract_features.py    # CFG extraction with angr CFGFast
│   ├── tokenizer.py           # Domain-specific assembly tokenizer (vocab_size=5000)
│   └── batch_preprocess.py    # Batch processing for Dataset-1
├── dataset/                    # US2: PyTorch data loading
│   ├── code_dataset.py        # Dynamic pairing from metadata.csv
│   └── collate.py             # Heterogeneous batch collation (sequences + graphs)
├── models/                     # US3-US5: Neural architecture
│   ├── gnn_encoder.py         # GAT encoder (3 layers, 4 heads, configurable output_dim)
│   ├── custom_attention.py    # KV-Prefix attention mechanism (THE HARD PART)
│   ├── bert_encoder.py        # BERT with graph prefix injection
│   └── bcsd_model.py          # Integrated model (GNN→BERT)
├── training/                   # US7: Training infrastructure
│   ├── losses.py              # MLM, InfoNCE, Joint losses
│   ├── metrics.py             # Validation metrics
│   └── trainer.py             # Training loop with checkpointing
├── inference/                  # US8: Vectorization & search
│   ├── vectorizer.py          # Binary → embedding extraction
│   └── similarity.py          # Cosine similarity search
├── utils/                      # Cross-cutting concerns
│   ├── logging.py             # Structured logging
│   └── reproducibility.py     # Seed management for determinism
├── configs/                    # Configuration files
│   ├── model_config.yaml      # GNN/BERT architecture params
│   └── train_config.yaml      # Training hyperparameters
├── data/                       # Generated data (gitignored)
│   ├── vocab.json             # Assembly vocabulary
│   ├── metadata.csv           # Preprocessed binary metadata
│   └── preprocessed/          # {hash}.json files per binary
├── test_binaries/             # Validation test suite
│   ├── test_gnn.c             # Simple C program for validation
│   ├── compile.sh             # Compilation script (gcc O0/O3 variants)
│   └── expected_outputs/      # Expected CFG patterns (documentation)
├── tests/                      # Unit tests (pytest)
│   ├── test_preprocessing.py
│   ├── test_dataset.py
│   ├── test_gnn.py
│   ├── test_bert_integration.py
│   ├── test_training.py
│   └── test_inference.py
├── protocols/                  # US9: Research session logs
│   └── session_YYYY-MM-DD.md  # Daily session documentation
├── thesis/                     # US9: LaTeX thesis document
│   ├── main.tex               # Thesis main file
│   └── methodology.tex        # Auto-generated methodology chapter
├── scripts/                    # Entry points and utilities
│   ├── train_model.py         # Training entry point
│   ├── run_inference.py       # Inference entry point
│   ├── start_session.sh       # US9: Start research session
│   ├── end_session.sh         # US9: End session with LaTeX extraction
│   └── review_document.py     # US10: Document critique agent
├── requirements.txt            # Pinned dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

- **Python 3.11+** (recommended: 3.11 or 3.12)
- **CUDA 11.8+** (for GPU training)
- **16GB RAM minimum** (32GB recommended for full Dataset-1 training)
- **6GB GPU VRAM minimum** (for batch_size=16)
- **Linux (Ubuntu/Debian)** - angr works best on Linux

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Nguyen-Bang/BCSD-Model-using-GNN-to-enrichen-V-K-.git
cd BCSD-Model-using-GNN-to-enrichen-V-K-
```

2. **Create virtual environment (recommended):**
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python -c "import angr, torch, torch_geometric, transformers; print('✓ All dependencies installed')"
```

## Quick Start

### Step 1: Validate Preprocessing on Test Binaries

Start with the controlled test binaries to verify the pipeline works:

```bash
# Compile test binaries (gcc O0 and O3 variants)
cd test_binaries
bash compile.sh
cd ..

# Preprocess test binaries
python preprocessing/batch_preprocess.py \
  --binary_dir test_binaries \
  --output_dir data/test_output \
  --vocab_file data/vocab.json
```

**Expected Output:**
- `data/vocab.json` - Assembly vocabulary (up to 5000 tokens)
- `data/test_output/metadata.csv` - Metadata for all processed binaries
- `data/test_output/{hash}.json` - One JSON file per binary with tokenized instructions and CFG edges

### Step 2: Preprocess Full Dataset (Dataset-1)

```bash
# Process all binaries in Dataset-1 (clamav, curl, nmap, openssl, unrar, z3, zlib)
python preprocessing/batch_preprocess.py \
  --binary_dir Dataset-1 \
  --output_dir data/preprocessed \
  --vocab_file data/vocab.json \
  --timeout 600
```

**Note:** This will take several hours for the full dataset. Progress is logged to console and `preprocessing.log`.

### Step 3: Train the Model

```bash
# Train with default configuration (MLM + Contrastive loss, λ=0.5)
python scripts/train_model.py \
  --config configs/train_config.yaml \
  --metadata data/preprocessed/metadata.csv \
  --vocab data/vocab.json \
  --output_dir checkpoints/
```

**Training Configuration (`configs/train_config.yaml`):**
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 10
- Loss weight λ: 0.5 (balance MLM vs contrastive)
- Early stopping patience: 3 epochs

**Expected Outputs:**
- `checkpoints/model_epoch_{N}_valloss_{loss}.pt` - Model checkpoints
- `logs/training_metrics.csv` - Epoch-level metrics (train_loss, val_loss, mlm_loss, contrastive_loss)

### Step 4: Vectorize Binaries and Search

```bash
# Vectorize all test binaries
python scripts/run_inference.py \
  --checkpoint checkpoints/best_model.pt \
  --binary_dir test_binaries \
  --output_dir embeddings/

# Find similar binaries (query with test_gnn_gcc_O0, should retrieve test_gnn_gcc_O3 as top match)
python scripts/run_inference.py \
  --checkpoint checkpoints/best_model.pt \
  --query test_binaries/test_gnn_gcc_O0 \
  --database embeddings/ \
  --top_k 5
```

**Expected Output:**
- Cosine similarity scores for top-K similar binaries
- `test_gnn_gcc_O3` should appear as top match (same function, different optimization level)

## Usage Guide

### Preprocessing

**Single Binary:**
```python
from preprocessing.extract_features import extract_cfg
from preprocessing.tokenizer import AssemblyTokenizer

# Extract CFG
result = extract_cfg("path/to/binary", output_dir="data/preprocessed")

# Tokenize
tokenizer = AssemblyTokenizer(vocab_size=5000, max_seq_length=512)
tokenizer.load_vocab("data/vocab.json")
tokens = tokenizer.tokenize(result["instructions"])
```

**Batch Processing:**
```bash
# Process entire directory
python preprocessing/batch_preprocess.py --binary_dir Dataset-1/clamav --output_dir data/preprocessed
```

### Training

**From Config File:**
```python
from training.trainer import Trainer
from models.bcsd_model import BCSModel

model = BCSModel(graph_dim=256, bert_model="bert-base-uncased")
trainer = Trainer(model, config="configs/train_config.yaml")
trainer.train(metadata_csv="data/preprocessed/metadata.csv", epochs=10)
```

**Custom Training Loop:**
```python
from dataset.code_dataset import BinaryCodeDataset
from torch.utils.data import DataLoader
from dataset.collate import collate_heterogeneous

# Load dataset with dynamic pairing
dataset = BinaryCodeDataset(metadata_csv="data/preprocessed/metadata.csv", split="train")
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_heterogeneous)

# Training loop
for batch in dataloader:
    embeddings = model(batch["input_ids"], batch["attention_mask"], batch["graph_batch"])
    loss = compute_joint_loss(embeddings, batch["labels"])
    loss.backward()
    optimizer.step()
```

### Inference

**Vectorize Single Binary:**
```python
from inference.vectorizer import Vectorizer

vectorizer = Vectorizer(checkpoint_path="checkpoints/best_model.pt")
embedding = vectorizer.vectorize_binary("path/to/binary")  # Returns [768,] numpy array
```

**Similarity Search:**
```python
from inference.similarity import cosine_similarity, top_k_similar
import numpy as np

# Load embedding database
database = np.load("embeddings/database.npy")
query_embedding = vectorizer.vectorize_binary("query_binary")

# Find top-5 similar
results = top_k_similar(query_embedding, database, k=5)
print(f"Top match: {results[0]['binary_path']} (score: {results[0]['similarity']:.4f})")
```

## Architecture Details

### 1. Preprocessing Pipeline (`preprocessing/`)

**Extract CFG with angr:**
- Uses `CFGFast` analysis (fast, accurate, no symbolic execution overhead)
- Extracts nodes (basic blocks with instructions), edges (control flow), function metadata
- Handles timeouts, corrupted binaries, library code filtering

**Assembly Tokenization:**
- Custom domain-specific tokenizer (NOT BERT's WordPiece)
- Vocabulary built from opcodes, registers, immediates (up to 5000 tokens)
- Special tokens: `[PAD]=0`, `[CLS]=101`, `[SEP]=102`, `[MASK]=103`, `[UNK]=104`
- Padding/truncation to `max_seq_length=512`

### 2. Data Loading (`dataset/`)

**Dynamic Pairing from metadata.csv:**
- Groups binaries by `(project, function_name)` for positive pairs
- Randomly samples 2 binaries from same function group per batch
- Heterogeneous batching: pads sequences, batches graphs with PyTorch Geometric

### 3. Neural Architecture (`models/`)

**GNN Encoder (GAT):**
- 3 GAT layers, 4 attention heads, LeakyReLU, dropout=0.2
- Configurable `output_dim` ∈ {128, 256, 512} for experimentation
- Global attention pooling: variable-size CFG → fixed-size summary `[batch, graph_dim]`

**KV-Prefix Attention (THE INNOVATION):**
- Projects `graph_summary` → `prefix_k` and `prefix_v` using separate linear layers: `graph_dim → 768`
- Reshapes for multi-head attention: `[batch, 12 heads, 1, 64 head_dim]`
- Concatenates prefix to sequence K/V along length dimension: `[batch, heads, seq_len+1, head_dim]`
- Extends attention mask to always attend to prefix position
- Injected into ALL 12 BERT layers (deep integration)

**BERT Integration:**
- Loads `bert-base-uncased` (12 layers, 768 hidden, 12 heads)
- Replaces standard attention with `KVPrefixAttention` in all layers
- Forward pass: token embeddings → 12 layers with graph prefix → `[CLS]` embedding extraction

### 4. Training (`training/`)

**Joint Loss:**
- **MLM Loss:** Masked language modeling on assembly instructions (predict masked tokens)
- **InfoNCE Loss:** Contrastive learning with in-batch negatives, temperature=0.07
- **Combined:** `total_loss = mlm_loss + λ * contrastive_loss` where λ ∈ {0.3, 0.5, 0.7} configurable

**Training Loop:**
- Siamese training: same model processes all samples in batch sequentially
- AdamW optimizer, lr=2e-5, gradient clipping
- Validation every epoch, early stopping (patience=3)
- Checkpointing: `model_epoch_{N}_valloss_{loss}.pt`
- Epoch-level logging to CSV: `train_loss`, `val_loss`, `mlm_loss`, `contrastive_loss`

### 5. Inference (`inference/`)

**Vectorization:**
- Loads trained model checkpoint → eval mode
- Preprocesses binary → forward pass → extracts `[CLS]` token embedding `[768,]`
- Batch processes directories → saves embeddings as `.npy` files

**Similarity Search:**
- Cosine similarity: `similarity = (A · B) / (||A|| * ||B||)`
- Numpy-based (no GPU needed), <1 second for 10,000 vectors
- Returns top-K similar binaries with scores

## Thesis Research Support

### Session Management (US9)

Document research activities with automatic LaTeX integration:

```bash
# Start new research session
bash scripts/start_session.sh

# ... work on experiments, log findings in protocols/session_YYYY-MM-DD.md ...

# End session and extract key points to thesis/methodology.tex
bash scripts/end_session.sh
```

**Session Template:**
- Goals: What to accomplish today
- Actions: Commands run, experiments conducted
- Outcomes: Results, metrics, observations
- Issues: Problems encountered, solutions attempted

### Document Review (US10)

Get automated feedback on thesis clarity:

```bash
# Review methodology chapter
python scripts/review_document.py thesis/methodology.tex

# Get feedback on specific section
python scripts/review_document.py thesis/methodology.tex --section 3.2
```

**Critique Areas:**
- Paragraph flow and transitions
- Technical clarity (jargon, definitions)
- Sentence complexity and readability
- Repetition and redundancy

## Troubleshooting

### angr CFG Extraction Fails

**Symptoms:** `extract_features.py` hangs or throws errors
**Solutions:**
1. Increase timeout: `--timeout 1200` (20 minutes)
2. Check binary format: `file path/to/binary` (must be ELF for Linux)
3. Test on simple binary first: `test_binaries/test_gnn_gcc_O0`

### Out of Memory During Training

**Symptoms:** `CUDA out of memory` error
**Solutions:**
1. Reduce batch size in `configs/train_config.yaml`: `batch_size: 8`
2. Reduce `max_seq_length` in tokenizer: `max_seq_length: 256`
3. Use smaller `graph_dim` in `configs/model_config.yaml`: `graph_dim: 128`

### Preprocessing Too Slow

**Symptoms:** `batch_preprocess.py` takes too long
**Solutions:**
1. Process subsets: Filter by project in `Dataset-1/`
2. Use parallel processing: Multiple instances with different `--binary_dir` subdirectories
3. Skip validation binaries initially: Process train/val splits first

### Model Not Learning (Loss Not Decreasing)

**Symptoms:** Validation loss stays flat or increases
**Solutions:**
1. Check loss weights: Try different λ values (0.3, 0.5, 0.7)
2. Verify data quality: Inspect `data/preprocessed/metadata.csv` for errors
3. Reduce learning rate: `learning_rate: 1e-5` in train_config.yaml
4. Check for data leakage: Ensure test functions not in training set

## Citation

If you use this code for research, please cite:

```bibtex
@misc{nguyen2025bcsd,
  author = {Nguyen, Bang},
  title = {Binary Code Similarity Detection Using GNN-Enriched BERT with KV-Prefix Attention},
  year = {2025},
  howpublished = {\url{https://github.com/Nguyen-Bang/BCSD-Model-using-GNN-to-enrichen-V-K-}},
}
```

## License

This project is part of a Bachelor's thesis research. See LICENSE file for details.

## Contributing

This is a research project under active development. Issues and pull requests are welcome, but please note this is primarily for academic purposes.

## Contact

- **Author:** Nguyen Bang
- **GitHub:** [@Nguyen-Bang](https://github.com/Nguyen-Bang)
- **Repository:** [BCSD-Model-using-GNN-to-enrichen-V-K-](https://github.com/Nguyen-Bang/BCSD-Model-using-GNN-to-enrichen-V-K-)

## Acknowledgments

- angr binary analysis framework
- Hugging Face Transformers library
- PyTorch and PyTorch Geometric teams
- Dataset-1 binary corpus contributors

# Initialize pipeline
pipeline = BCSDPipeline(bert_model='bert-base-uncased', device='cuda')

# Process a single binary
result = pipeline.process_single_binary('/path/to/binary')

# Process multiple binaries
results = pipeline.process_binary_folder('/path/to/binaries')

# Compare two binaries
similarity = pipeline.compare_binaries(result1, result2)
print(f"Similarity: {similarity['overall_similarity']:.4f}")

# Find similar binaries
similar_pairs = pipeline.find_similar_binaries(results, threshold=0.7)
```

## Technical Details

### Architecture

1. **Disassembly Layer**: Uses angr to extract control flow graphs and assembly instructions
2. **Encoding Layer**: BERT transforms instructions into 768-dimensional embeddings
3. **Graph Layer**: GNN processes the graph structure and produces enriched embeddings (default: 128-dimensional)
4. **Similarity Layer**: Cosine similarity computation between enriched embeddings

### Model Configuration

The GNN model can be configured with:
- `input_dim`: Dimension of input features (default: 768 for BERT)
- `hidden_dim`: Dimension of hidden layers (default: 256)
- `output_dim`: Dimension of output embeddings (default: 128)
- `num_layers`: Number of GNN layers (default: 3)
- `dropout`: Dropout rate (default: 0.1)
- `gnn_type`: Type of GNN ('GCN' or 'GAT')

## Requirements

See `requirements.txt` for a complete list of dependencies. Main requirements:
- angr >= 9.2.0
- transformers >= 4.30.0
- torch >= 2.0.0
- torch-geometric >= 2.3.0

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bcsd_gnn_model,
  title = {BCSD Model using GNN to enrich BERT embeddings},
  author = {Nguyen-Bang},
  year = {2024},
  url = {https://github.com/Nguyen-Bang/BCSD-Model-using-GNN-to-enrichen-V-K-}
}
```