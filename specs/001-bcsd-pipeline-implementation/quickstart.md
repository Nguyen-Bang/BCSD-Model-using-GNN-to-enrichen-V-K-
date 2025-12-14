# Quickstart Guide

**Feature**: Complete BCSD Pipeline Implementation  
**Date**: 2025-12-13  
**Audience**: Developers and researchers setting up the BCSD system

---

## Prerequisites

### System Requirements

**Hardware**:
- CPU: Modern x86_64 processor (Intel/AMD)
- RAM: Minimum 16GB, recommended 32GB
- GPU: NVIDIA GPU with 6GB+ VRAM (e.g., GTX 1060, RTX 2060, or better)
- Storage: 50GB free disk space (20GB for data, 30GB for models/checkpoints)

**Software**:
- OS: Ubuntu 20.04+ or Debian 11+ (Linux required for angr binary analysis)
- CUDA: 11.8 or compatible (for GPU support)
- Git: For version control
- Python: 3.11 or later

### Knowledge Prerequisites

- Basic Python programming
- Familiarity with PyTorch (tensors, nn.Module, DataLoader)
- Understanding of Git workflow
- Command-line proficiency (bash)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/BCSD-Model-using-GNN-to-enrichen-V-K-.git
cd BCSD-Model-using-GNN-to-enrichen-V-K-
```

### 2. Create Python Environment

**Option A: Using venv** (recommended)
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**Option B: Using conda**
```bash
conda create -n bcsd python=3.11
conda activate bcsd
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected installation time**: 5-10 minutes

**Key dependencies**:
- `angr==9.2.0` - Binary analysis framework
- `torch==2.0.1` - Deep learning framework
- `torch-geometric==2.3.0` - GNN library
- `transformers==4.30.0` - BERT models
- `networkx==3.1` - Graph data structures
- `matplotlib==3.7.0` - Visualization
- `jupyter==1.0.0` - Interactive notebooks

### 4. Verify GPU Support (Optional but Recommended)

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Architecture Overview

**Phase-Based Pipeline Structure**:

```text
PHASE 1: Data Generation (preprocessing/)
  ├── extract_features.py → CFG + Disassembly extraction via angr
  └── tokenizer.py → Custom assembly tokenization (vocab, masking)

PHASE 2: Data Loading (dataset/)
  ├── code_dataset.py → PyTorch Dataset
  ├── collate.py → Heterogeneous batching
  └── pairing.py → Siamese pair generation

PHASE 3: Neural Architecture (models/)
  ├── gnn_encoder.py → GAT for graph summarization
  ├── custom_attention.py → KV-Prefix attention (HARD PART)
  ├── bert_encoder.py → BERT with custom attention
  └── bcsd_model.py → God class: GNN→Projection→BERT

PHASE 4: Training (training/)
  ├── trainer.py → Training loop
  ├── losses.py → MLM + InfoNCE contrastive
  └── metrics.py → Validation metrics

PHASE 5: Inference (inference/)
  ├── vectorizer.py → Binary → Embedding
  ├── similarity.py → Cosine similarity search
  └── clustering.py → Optional clustering
```

**Key Insight**: Each phase is independently testable. This structure follows the data flow from raw binaries to similarity detection.

Expected output: `CUDA available: True`

If False, you can still run on CPU (training will be much slower).

---

## Dataset Setup

### 1. Obtain Dataset-1

Download the binary datasets (Dataset-1) containing:
- **Training**: clamav, curl, nmap, openssl
- **Validation**: unrar
- **Test**: z3, zlib

Place binaries in:
```
Dataset-1/
├── clamav/
│   ├── clamscan
│   ├── clamd
│   └── ...
├── curl/
│   ├── curl
│   └── ...
├── nmap/
├── openssl/
├── unrar/
├── z3/
└── zlib/
```

### 2. Create Data Directory Structure

```bash
mkdir -p data/preprocessed
mkdir -p embeddings/graph_summaries
mkdir -p embeddings/binary_embeddings
mkdir -p checkpoints
mkdir -p logs
mkdir -p protocols
```

### 3. Index Binaries

Create `data/binaries_index.csv` with all binary files:

```bash
python scripts/create_binary_index.py --dataset-dir Dataset-1 --output data/binaries_index.csv
```

This script:
- Scans Dataset-1 directory
- Computes SHA256 hashes
- Assigns train/validation/test splits
- Creates CSV with metadata

**Expected output**: `data/binaries_index.csv` with ~100-200 entries

---

## Quick Test Run

### 0. Validate Disassembly with Test Binary (FIRST STEP)

**Purpose**: Validate angr disassembly and tokenization on a controlled test case before processing the full dataset. This provides a concrete example for thesis documentation.

```bash
# Compile test binary with multiple variants
cd test_binaries
bash compile.sh
cd ..

# Extract CFG and instructions from test binary
python -c "
from pipeline.angr_disassembly import extract_function_data
import json

# Process gcc O0 variant
result = extract_function_data('test_binaries/test_gnn_gcc_O0')
print(json.dumps(result, indent=2))

# Save to file for inspection
with open('test_binaries/test_gnn_gcc_O0_output.json', 'w') as f:
    json.dump(result, f, indent=2)
"
```

**Expected output structure**:
```json
{
  "functions": [
    {
      "function_name": "calculate_sum",
      "function_address": "0x401130",
      "blocks": [
        {"id": 0, "address": "0x401130", "instructions": ["push rbp", "mov rbp rsp", ...]},
        {"id": 1, "address": "0x401140", "instructions": ["cmp eax edi", ...]},
        {"id": 2, "address": "0x401148", "instructions": ["add eax 0x1", ...]}
      ],
      "edges": [[0, 1], [1, 2], [2, 1], [1, 3]]  // Note: [2,1] is loop backward edge
    },
    {
      "function_name": "check_value",
      "blocks": [...],
      "edges": [[0, 1], [0, 2], [1, 3], [2, 3]]  // Multiple paths from branches
    },
    {
      "function_name": "main",
      "blocks": [...],
      "edges": [...]
    }
  ]
}
```

**Manual validation steps**:
1. Open `test_binaries/test_gnn_gcc_O0_output.json`
2. Verify `calculate_sum` contains backward edge (loop cycle)
3. Verify `check_value` contains branch nodes (multiple outgoing edges)
4. Verify instruction tokenization separates opcodes from operands
5. Compare with clang/O3 variants to see optimization effects
6. Use this output as running example in thesis methodology chapter

**Processing time**: ~10 seconds

### 1. Preprocess a Single Binary (Test on Real Data)

```bash
python -c "
from pipeline.angr_disassembly import extract_cfg
result = extract_cfg(
    binary_path='Dataset-1/curl/curl',
    output_dir='data/preprocessed',
    timeout=300
)
print('Status:', result['status'])
print('Nodes:', result.get('node_count', 'N/A'))
print('Edges:', result.get('edge_count', 'N/A'))
"
```

**Expected output**:
```
Status: success
Nodes: 1523
Edges: 2048
```

**Processing time**: ~1-2 minutes for curl binary

### 2. Test Dataset Loading

```bash
python -c "
from pipeline.dataset import BCSDataset
dataset = BCSDataset(data_dir='data', split='train')
print(f'Dataset size: {len(dataset)} samples')
sample = dataset[0]
print(f'Sample keys: {sample.keys()}')
"
```

**Expected output**:
```
Dataset size: 150 samples
Sample keys: dict_keys(['binary_hash', 'cfg', 'instruction_tokens', 'metadata'])
```

### 3. Test Model Initialization

```bash
python -c "
from pipeline.gnn_model import BCSModel
model = BCSModel(
    gnn_config={'input_dim': 768, 'hidden_dim': 256, 'output_dim': 256},
    bert_config={'bert_model_name': 'bert-base-uncased', 'graph_dim': 256}
)
print('Model parameters:', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
"
```

**Expected output**:
```
Model parameters: 110.5 M
```

---

## Full Pipeline Execution

### 1. Preprocess All Binaries

```bash
python scripts/preprocess_dataset.py \
    --index-file data/binaries_index.csv \
    --output-dir data/preprocessed \
    --workers 4
```

**Expected time**: 2-4 hours for full dataset (depends on binary sizes)

**Output**: JSON files in `data/preprocessed/` for each binary

### 2. Generate Training Pairs

```bash
python scripts/generate_training_pairs.py \
    --index-file data/binaries_index.csv \
    --output-dir data \
    --num-negatives 3
```

**Expected output**:
- `data/training_pairs_train.csv`
- `data/training_pairs_validation.csv`

### 3. Train Model

```bash
python scripts/train.py \
    --config configs/train_config.yaml \
    --data-dir data \
    --checkpoint-dir checkpoints \
    --log-dir logs
```

**Training config** (`configs/train_config.yaml`):
```yaml
epochs: 10
batch_size: 16
learning_rate: 2.0e-5
weight_decay: 0.01
lambda_contrastive: 0.5
stage1_epochs: 5
stage2_epochs: 5
checkpoint_dir: checkpoints/
log_dir: logs/
seed: 42
```

**Expected time**: 12-24 hours on single GPU (RTX 3060)

**Monitoring**: Training logs written to `logs/training_metrics.csv`

### 4. Run Inference

```bash
python scripts/inference.py \
    --binary Dataset-1/z3/z3 \
    --checkpoint checkpoints/model_epoch_10_valloss_0.2345.pt \
    --embedding-db embeddings/embedding_database.pkl \
    --top-k 10
```

**Expected output**:
```
Top 10 Similar Binaries:
1. z3_optimizer (score: 0.8543)
2. z3_solver (score: 0.8201)
3. zlib_compress (score: 0.7856)
...
```

---

## Exploration Notebook

### Launch Jupyter

```bash
jupyter notebook exploration.ipynb
```

This opens an interactive notebook with 4 phases:
1. **Phase 1**: Visualize CFG from preprocessed data
2. **Phase 2**: Test GNN encoder on sample graphs
3. **Phase 3**: Validate custom attention mechanism
4. **Phase 4**: End-to-end forward pass with complete model

**Usage**: Execute each cell sequentially. Each phase is independent.

---

## Session Management

### Start Research Session

```bash
./scripts/start_session.sh
```

Creates `protocols/session_YYYY-MM-DD.md` with template:
```markdown
# Session: 2025-12-13

## Goals
- [ ] Goal 1
- [ ] Goal 2

## Actions Taken
- 10:00 - Started preprocessing curl binaries
- 10:30 - Debugged shape mismatch in collate_fn

## Outcomes
- Successfully preprocessed 50 binaries
- Fixed padding issue in DataLoader

## Issues Encountered
- Out of memory with batch_size=32, reduced to 16
```

### End Research Session

```bash
./scripts/end_session.sh
```

Prompts for summary and appends key learnings to `thesis/methodology.tex`.

---

## Common Issues and Solutions

### Issue: angr Timeout on Large Binaries

**Symptom**: Preprocessing takes >10 minutes per binary

**Solution**: Increase timeout parameter:
```python
extract_cfg(binary_path, output_dir, timeout=1200)  # 20 minutes
```

---

### Issue: CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size in config: `batch_size: 8`
2. Enable gradient checkpointing (reduces memory, increases time)
3. Use CPU (slow but works): `device: "cpu"`

---

### Issue: Empty CFG Extracted

**Symptom**: `node_count: 0` in preprocessing result

**Possible causes**:
- Binary is stripped (symbols removed)
- Corrupted binary file
- Unsupported architecture

**Solution**: Check binary with `file` command, ensure it's ELF x86_64

---

### Issue: DataLoader Shape Mismatch

**Symptom**: `RuntimeError: shape mismatch in collate_fn`

**Solution**: Verify all preprocessed files have valid structure:
```bash
python scripts/validate_preprocessed_data.py --data-dir data/preprocessed
```

---

## Validation Checklist

Before starting full training, verify:

- [ ] GPU is accessible (`torch.cuda.is_available() == True`)
- [ ] At least 10 binaries preprocessed successfully
- [ ] Dataset loading works without errors
- [ ] Model initializes and forward pass succeeds on dummy data
- [ ] Checkpoints directory is writable
- [ ] At least 20GB free disk space for checkpoints

---

## Next Steps

After successful quickstart:

1. **Experiment Tracking**: Log all experiments in `protocols/` directory
2. **Hyperparameter Tuning**: Try different learning rates, batch sizes
3. **Baseline Comparisons**: Train BERT-only and GNN-only baselines
4. **Evaluation**: Run full evaluation on test set (z3, zlib)
5. **Thesis Writing**: Document findings in LaTeX thesis document

---

## Getting Help

**Documentation**:
- `/specs/001-bcsd-pipeline-implementation/research.md` - Technical decisions
- `/specs/001-bcsd-pipeline-implementation/data-model.md` - Data structures
- `/specs/001-bcsd-pipeline-implementation/contracts/` - API specifications

**Debugging**:
- Check logs in `logs/` directory
- Enable debug logging: `export LOG_LEVEL=DEBUG`
- Run with `--verbose` flag for detailed output

**Code Review**:
- Use `scripts/review_code.py` for automated checks
- Run pytest: `pytest tests/`

---

## Reproducibility Notes

To ensure reproducible results:

1. **Set seed**: All scripts use `seed=42` from config
2. **Pin dependencies**: Use exact versions from `requirements.txt`
3. **Document hardware**: Note GPU model and CUDA version
4. **Version control**: Commit after each major milestone
5. **Log git hash**: Training scripts automatically log commit hash

**Expected variance**: Results should be reproducible within 5% (as per SC-016).
