# Data Model

**Feature**: Complete BCSD Pipeline Implementation  
**Date**: 2025-12-13  
**Status**: Phase 1 - Data Model Design

## Overview

This document defines the core entities, their attributes, relationships, and validation rules for the BCSD pipeline. The data model supports the flow: Binary → CFG → GNN Embeddings → BERT Integration → Similarity Scores.

---

## Entity Definitions

### 1. BinaryExecutable

**Description**: Represents an ELF binary file to be analyzed.

**Attributes**:
- `file_path` (str, required): Absolute path to the binary file
- `file_hash` (str, required): SHA256 hash for unique identification (used as filename prefix)
- `project` (str, required): Source dataset name (e.g., "clamav", "curl", "nmap", "openssl", "unrar", "z3", "zlib")
- `function_name` (str, required): Name of the function extracted from this binary
- `optimization` (str, required): Optimization level ("O0", "O1", "O2", "O3")
- `compiler` (str, required): Compiler used ("gcc", "clang")
- `split_set` (str, required): Dataset split ("train", "validation", "test")
- `file_size_bytes` (int, required): Size of binary file in bytes
- `architecture` (str, optional): Target architecture (e.g., "x86_64", "arm")
- `is_stripped` (bool, required): Whether symbols are stripped
- `preprocessing_status` (str, required): Status of CFG extraction ("pending", "success", "failed")
- `preprocessing_error` (str, optional): Error message if preprocessing failed
- `created_at` (datetime, required): Timestamp when record was created

**Validation Rules**:
- `file_path` must exist and be readable
- `file_hash` must be unique across all binaries (SHA256 ensures uniqueness)
- `project` must be one of the 7 valid datasets
- `split_set` assignment: {clamav, curl, nmap, openssl} → "train", {unrar} → "validation", {z3, zlib} → "test"
- `optimization` must be one of: "O0", "O1", "O2", "O3"
- `compiler` must be one of: "gcc", "clang"
- `file_size_bytes` > 0
- `preprocessing_status` transitions: pending → success|failed (one-way)

**Relationships**:
- One-to-One with unified JSON file in `data/preprocessed/{file_hash}.json`
- One-to-One with `Embedding` (after model inference)

**File Structure**:
- Stored in: `data/metadata.csv` (master index)
- Columns: file_hash, project, function_name, optimization, compiler, split_set, file_path, file_size_bytes, architecture, is_stripped, preprocessing_status, created_at

---

### 2. PreprocessedFunction

**Description**: Unified representation of a single function combining CFG structure and tokenized sequence in one JSON file. This replaces the previous separate {hash}_cfg.json and {hash}_seq.json files.

**File Location**: `data/preprocessed/{file_hash}.json`

**JSON Schema**:
```json
{
  "id": "a1b2c3d4e5f6...",           // SHA256 hash (matches file_hash in metadata.csv)
  "function_name": "ssl_read",       // Function identifier
  "tokens": ["MOV", "EAX", "EBX", ...],  // Tokenized instruction sequence
  "token_count": 245,                 // Number of tokens
  "edges": [[0, 1], [1, 2], [2, 0], ...], // CFG edge list (node_id pairs)
  "node_count": 128,                  // Number of basic blocks (CFG nodes)
  "edge_count": 156                   // Number of CFG edges
}
```

**Attributes**:
- `id` (str, required): SHA256 hash matching file_hash in metadata
- `function_name` (str, required): Function name
- `tokens` (List[str], required): Tokenized assembly instructions
- `token_count` (int, required): Length of token sequence
- `edges` (List[List[int]], required): CFG edge list [[source, target], ...]
- `node_count` (int, required): Number of nodes in CFG
- `edge_count` (int, required): Number of edges in CFG

**Validation Rules**:
- `id` must match filename (without .json extension)
- `tokens` must not be empty
- `token_count` == len(tokens)
- `edges` format: list of [source_id, target_id] pairs where IDs are in range [0, node_count-1]
- `edge_count` == len(edges)
- No self-loops in edges (source_id != target_id)

**Performance Benefit**: Single file read per function (2x faster than separate cfg+seq files), reduces I/O bottleneck during training.

**Relationships**:
- Referenced by file_hash in metadata.csv
- Loaded directly by PyTorch Dataset class

---

### 3. ControlFlowGraph (CFG) - DEPRECATED

**Note**: This entity is now merged into PreprocessedFunction. Kept for reference only.

**Description**: Directed graph representing control flow structure extracted from a binary.

**NEW STRUCTURE**: CFG data is now stored within `data/preprocessed/{file_hash}.json` as the `edges`, `node_count`, and `edge_count` fields.

---

### 4. InstructionSequence - DEPRECATED

**Note**: This entity is now merged into PreprocessedFunction. Kept for reference only.

**Description**: Linearized sequence of assembly instructions for BERT input.

**NEW STRUCTURE**: Token sequence is now stored within `data/preprocessed/{file_hash}.json` as the `tokens` and `token_count` fields.

---

### 5. GraphSummary

**Description**: Fixed-size vector representation of a CFG produced by GNN encoder.

**Attributes**:
- `file_hash` (str, required, foreign key): References file_hash in metadata.csv
- `vector` (np.ndarray, required): 256-dimensional float array (configurable: 128, 256, or 512)
- `dimension` (int, required): Dimensionality of the vector (experimental parameter)
- `pooling_method` (str, required): Global pooling method used ("mean", "attention", "max")
- `gnn_model_version` (str, required): Version/checkpoint of GNN model used
- `computed_at` (datetime, required): Timestamp of computation

**Validation Rules**:
- `vector.shape` == (dimension,)
- `vector` contains no NaN or Inf values
- `vector` is normalized (L2 norm close to 1.0, within 0.01 tolerance)
- `dimension` in {128, 256, 512} (experimental values)
- `pooling_method` must be one of valid methods

**Relationships**:
- Referenced by file_hash in metadata.csv
- Used as input to custom attention mechanism

---

### 6. Embedding

**Description**: Dense vector representation of a function in semantic space (from BERT).

**Attributes**:
- `file_hash` (str, required, foreign key): References file_hash in metadata.csv
- `vector` (np.ndarray, required): 768-dimensional float array (BERT hidden size)
- `dimension` (int, required): Dimensionality (default: 768)
- `source_layer` (str, required): Which BERT output was used ("cls_token", "mean_pooling", "last_hidden")
- `model_checkpoint` (str, required): Path to model checkpoint used for inference
- `computed_at` (datetime, required): Timestamp of computation

**Validation Rules**:
- `vector.shape` == (dimension,)
- `vector` contains no NaN or Inf values
- `vector` is normalized (L2 norm close to 1.0)
- `dimension` == 768 for bert-base
- `source_layer` must be one of valid options

**Relationships**:
- Referenced by file_hash in metadata.csv
- Used for similarity computation

---

### 7. TrainingSample (Dynamic Positive Pairs) - NO PRE-COMPUTED CSV

**Description**: Positive pairs are generated dynamically during training by the PyTorch Dataset class, not pre-computed and saved to CSV. This enables flexible pairing strategies (e.g., epoch 1 pairs O0+O3, epoch 2 pairs O1+O2).

**Dynamic Pairing Strategy**:
1. Dataset class reads `data/metadata.csv` and filters by `split_set == "train"`
2. Groups functions by `(project, function_name)` - e.g., all versions of `openssl::ssl_read`
3. At runtime (`__getitem__`), randomly samples two file_hashes from the same group
4. Loads corresponding `data/preprocessed/{hash_A}.json` and `data/preprocessed/{hash_B}.json`
5. Returns: (tokens_A, edges_A, tokens_B, edges_B, label=1)

**In-Batch Negatives**:
- NOT stored explicitly
- All other samples in batch serve as negatives
- For anchor `ssl_read_gcc_O0` with positive `ssl_read_clang_O3`, negatives are other functions in batch (e.g., `parse_gcc_O0`, `search_clang_O3`)

**Attributes** (computed at runtime, not stored):
- `file_hash_A` (str): First function variant
- `file_hash_B` (str): Second function variant  
- `function_name` (str): Shared function name
- `label` (int): Always 1 (positive pair)

**Validation Rules** (enforced by Dataset class):
- `file_hash_A` != `file_hash_B` (no self-pairs)
- Both hashes belong to same `(project, function_name)` group
- Both hashes exist in `data/preprocessed/` directory

**Why No CSV**:
- **Flexibility**: Can change pairing strategy between epochs
- **Memory**: Avoids storing O(N^2) pairs for N functions
- **Simplicity**: Fewer files to manage

**Relationships**:
- Dynamically references two entries in metadata.csv
- Loads data from two PreprocessedFunction JSON files

---

### 8. ModelCheckpoint

**Description**: Saved state of the trained model after each epoch.

**Attributes**:
- `checkpoint_id` (str, required): Unique identifier (e.g., "epoch_05_valloss_0.23")
- `epoch` (int, required): Training epoch number
- `model_state_dict` (dict, required): PyTorch model state dictionary
- `optimizer_state_dict` (dict, required): Optimizer state (for resuming training)
- `train_loss` (float, required): Average training loss for this epoch
- `val_loss` (float, required): Validation loss for this epoch
- `mlm_loss` (float, required): Masked language modeling loss component (epoch average)
- `contrastive_loss` (float, required): Contrastive loss component (epoch average)
- `hyperparameters` (dict, required): All hyperparameters used (learning rate, batch size, lambda, etc.)
- `git_commit_hash` (str, required): Git commit hash at time of training
- `saved_at` (datetime, required): Timestamp when checkpoint was saved

**Validation Rules**:
- `epoch` >= 0
- `train_loss`, `val_loss`, `mlm_loss`, `contrastive_loss` >= 0.0
- `model_state_dict` must contain all required model parameters
- `hyperparameters` must include: learning_rate, batch_size, lambda_contrastive (∈ {0.3, 0.5, 0.7}), seed, graph_dim (∈ {128, 256, 512})
- `checkpoint_id` must be unique

**Training Logs CSV Schema** (`logs/training_metrics.csv`):
```
epoch, train_loss, val_loss, mlm_loss, contrastive_loss, total_loss
1, 2.34, 2.56, 1.87, 0.47, 2.34
2, 2.01, 2.23, 1.54, 0.47, 2.01
...
```

**Relationships**:
- Referenced by inference operations to load model

---

### 9. SessionLog

**Description**: Markdown document recording a research work session.

**Attributes**:
- `session_id` (str, required): Date-based identifier (e.g., "2025-12-13")
- `file_path` (str, required): Path to the Markdown file in `protocols/`
- `start_time` (datetime, required): When session started
- `end_time` (datetime, optional): When session ended (null if ongoing)
- `session_goals` (str, required): Goals for the session
- `actions_taken` (List[str], required): List of actions performed
- `outcomes` (List[str], optional): Results achieved
- `issues_encountered` (List[str], optional): Problems faced
- `key_learnings` (str, optional): Summary extracted for LaTeX document
- `latex_appended` (bool, required): Whether key learnings were added to thesis document

**Validation Rules**:
- `session_id` format: YYYY-MM-DD
- `file_path` must exist if session is ended
- `end_time` > `start_time` if both are set
- `key_learnings` required if `latex_appended` is True
- `actions_taken` must not be empty after session ends

**Relationships**:
- Generates content for LaTeX thesis document

---

### 10. TrainingMetrics

**Description**: Logged metrics during training (per step or per epoch).

**Attributes**:
- `log_id` (int, required): Unique identifier for this log entry
- `epoch` (int, required): Current epoch number
- `step` (int, required): Global training step
- `train_loss` (float, required): Total training loss
- `mlm_loss` (float, required): MLM component
- `contrastive_loss` (float, required): Contrastive component
- `val_loss` (float, optional): Validation loss (if validation step)
- `learning_rate` (float, required): Current learning rate (may decay)
- `timestamp` (datetime, required): When this metric was recorded

**Validation Rules**:
- `epoch` >= 0, `step` >= 0
- All loss values >= 0.0
- `learning_rate` > 0.0
- `val_loss` only present at validation steps
- `step` should monotonically increase

**Relationships**:
- Associated with a training run (implicitly linked via checkpoint_id)

---

## Data Flow

```
1. BinaryExecutable (input)
   ↓
2. ControlFlowGraph (angr preprocessing)
   ↓
3. BasicBlock (CFG nodes) + InstructionSequence (flattened tokens)
   ↓
4. GraphSummary (GNN encoder) + BERT token embeddings (parallel)
   ↓
5. Custom Attention (fusion)
   ↓
6. Embedding (final binary representation)
   ↓
7. TrainingSample (pairs for contrastive learning)
   ↓
8. ModelCheckpoint (trained model)
   ↓
9. Inference: BinaryExecutable → Embedding → Similarity Scores
```

---

## State Transitions

### BinaryExecutable.preprocessing_status
- Initial: `"pending"`
- On success: `"pending"` → `"success"`
- On failure: `"pending"` → `"failed"`
- Terminal states: `"success"`, `"failed"`

### SessionLog Lifecycle
- Created: `start_time` set, `end_time` = null, `latex_appended` = False
- During session: `actions_taken`, `outcomes`, `issues_encountered` updated
- Ended: `end_time` set, `key_learnings` extracted
- Finalized: `latex_appended` = True after LaTeX document updated

### ModelCheckpoint Evolution
- Each epoch creates new checkpoint
- Checkpoints are immutable after creation
- Best checkpoint selected based on `val_loss`

---

## Serialization Formats

| Entity | Format | Location | Purpose |
|--------|--------|----------|---------|
| BinaryExecutable | CSV | `data/binaries_index.csv` | Track all binaries and their metadata |
| ControlFlowGraph | JSON | `data/preprocessed/{binary_hash}_cfg.json` | Portable CFG representation |
| BasicBlock | JSON (nested in CFG) | Inside CFG files | Part of CFG structure |
| InstructionSequence | JSON | `data/preprocessed/{binary_hash}_seq.json` | Tokenized sequences |
| GraphSummary | NumPy (.npy) | `embeddings/graph_summaries/{binary_hash}.npy` | Fast loading for training |
| Embedding | NumPy (.npy) | `embeddings/binary_embeddings/{binary_hash}.npy` | Final embeddings |
| TrainingSample | CSV | `data/training_pairs_{split}.csv` | Generated pairs for training |
| ModelCheckpoint | PyTorch (.pt) | `checkpoints/model_epoch_{epoch}.pt` | Model state |
| SessionLog | Markdown (.md) | `protocols/session_{date}.md` | Human-readable logs |
| TrainingMetrics | CSV | `logs/training_metrics.csv` | Analysis-ready format |

---

## Indexing and Lookup

**Primary Keys**:
- BinaryExecutable: `binary_hash` (SHA256, 64 chars)
- ControlFlowGraph: `binary_hash` (same as binary)
- TrainingSample: `sample_id` (auto-increment integer)
- ModelCheckpoint: `checkpoint_id` (string)
- SessionLog: `session_id` (date string)

**Indexes for Fast Lookup**:
- BinaryExecutable.source_project (for splitting datasets)
- BinaryExecutable.split (train/val/test filtering)
- TrainingSample.label (positive/negative pair filtering)
- ModelCheckpoint.val_loss (finding best checkpoint)

---

## Data Validation Pipeline

1. **On Binary Ingestion**:
   - Verify file exists and is readable
   - Compute SHA256 hash
   - Check for duplicates in index
   - Assign to correct split based on source_project

2. **After CFG Extraction**:
   - Validate graph structure (no orphaned nodes, valid edge references)
   - Check for empty graphs (minimum 1 node)
   - Verify instruction sequences are non-empty

3. **Before Training**:
   - Verify all training binaries have both CFG and sequence data
   - Check for balanced positive/negative pairs
   - Validate tensor shapes match model expectations

4. **After Model Inference**:
   - Check embeddings for NaN/Inf values
   - Verify L2 normalization
   - Validate dimension consistency

---

## Error Handling

**Preprocessing Errors**:
- Corrupted binary → Mark as `failed`, log error in `preprocessing_error`
- Timeout (>10 minutes) → Mark as `failed`, skip to next binary
- Empty CFG → Mark as `failed`, retain binary in index for analysis

**Training Errors**:
- OOM (Out of Memory) → Reduce batch size, restart from last checkpoint
- Gradient explosion → Apply gradient clipping, log warning
- Diverging loss → Stop training, investigate hyperparameters

**Inference Errors**:
- Missing embedding → Recompute from checkpoint
- Shape mismatch → Validate model version matches preprocessing version

---

## Next Steps

1. Implement data loading utilities (`data_loader.py`)
2. Create validation functions for each entity
3. Define API contracts for module interfaces (next file: `contracts/`)
