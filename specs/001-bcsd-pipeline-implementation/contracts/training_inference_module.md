# Training and Inference Module Contract

**Module**: `pipeline.train` + `pipeline.inference`  
**Purpose**: Model training infrastructure and inference pipeline  
**Owner**: User Stories 7, 8 (P7-P8)

---

## Training Module API

### Function: `train_model(config: Dict, train_loader: DataLoader, val_loader: DataLoader, model: BCSModel, device: str) -> Dict[str, Any]`

**Description**: Complete Siamese training pipeline with checkpointing and logging.

**Siamese Training Strategy**:
- **Single Model Tower**: One model instance processes all samples
- **Batch Construction**: Each batch contains positive pairs (same function, different compilation variants)
- **Sequential Processing**: Model processes each sample individually with shared weights:
  ```python
  for (sample_A, sample_B) in positive_pairs:
      emb_A = model(sample_A)  # Same model weights
      emb_B = model(sample_B)  # Same model weights
      loss = contrastive_loss(emb_A, emb_B, in_batch_negatives)
  ```
- **In-Batch Negatives**: All other embeddings in the batch serve as negatives automatically

**Parameters**:
- `config` (Dict): Training configuration (see Config Schema below)
- `train_loader` (DataLoader): Training data with positive pairs
- `val_loader` (DataLoader): Validation data
- `model` (BCSModel): Single-tower model to train
- `device` (str): "cuda" or "cpu"

**Returns**: Training summary dictionary:
```python
{
    "best_checkpoint_path": str,
    "best_val_loss": float,
    "final_train_loss": float,
    "total_epochs": int,
    "training_time_seconds": float,
    "metrics_file": str  # Path to CSV with per-step metrics
}
```

**Behavior**:
1. Initializes optimizer (AdamW) and learning rate scheduler
2. **Stage 1 (MLM Pretraining, 5 epochs)**: Train on individual samples with MLM loss
3. **Stage 2 (Contrastive Fine-tuning, 5 epochs)**: Train on positive pairs with InfoNCE/NT-Xent loss
4. Validates after each epoch
5. Saves checkpoints with naming: `model_epoch_{epoch}_valloss_{loss:.4f}.pt`
6. Implements early stopping (patience=3 epochs)
7. Logs metrics to CSV file every N steps

---

### Config Schema

```python
{
    # Training hyperparameters
    "epochs": 10,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "gradient_clip_max_norm": 1.0,
    
    # Loss configuration
    "lambda_contrastive": 0.5,  # Weight for contrastive loss
    "mlm_mask_prob": 0.15,  # Probability of masking tokens
    "contrastive_temperature": 0.07,  # Temperature for InfoNCE/NT-Xent loss
    
    # Siamese training configuration
    "positive_pair_strategy": "function_level",  # Match same functions across compilations
    "in_batch_negatives": True,  # Use all other samples in batch as negatives
    
    # Two-stage training
    "stage1_epochs": 5,  # MLM pretraining (individual samples)
    "stage2_epochs": 5,  # Contrastive fine-tuning (positive pairs)
    "freeze_bert_in_stage1": True,
    
    # Checkpointing
    "checkpoint_dir": "checkpoints/",
    "save_every_n_epochs": 1,
    "keep_top_k_checkpoints": 3,  # Based on val_loss
    
    # Logging
    "log_dir": "logs/",
    "log_every_n_steps": 10,
    "tensorboard_enabled": False,
    
    # Early stopping
    "early_stopping_patience": 3,
    "early_stopping_delta": 0.001,
    
    # Reproducibility
    "seed": 42,
    "deterministic": True
}
```

---

### Function: `save_checkpoint(model: BCSModel, optimizer, epoch: int, metrics: Dict, config: Dict, checkpoint_path: str) -> None`

**Description**: Saves model checkpoint with all necessary state.

**Parameters**:
- `model` (BCSModel): Trained model
- `optimizer`: Optimizer state
- `epoch` (int): Current epoch number
- `metrics` (Dict): Training and validation metrics
- `config` (Dict): Training configuration
- `checkpoint_path` (str): Where to save checkpoint

**Checkpoint Structure**:
```python
{
    "epoch": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": dict,
    "train_loss": float,
    "val_loss": float,
    "mlm_loss": float,
    "contrastive_loss": float,
    "config": Dict,
    "git_commit_hash": str,
    "timestamp": str,
    "pytorch_version": str
}
```

---

### Function: `load_checkpoint(checkpoint_path: str, model: BCSModel, optimizer=None, device: str = "cuda") -> Dict`

**Description**: Loads checkpoint and restores training state.

**Parameters**:
- `checkpoint_path` (str): Path to checkpoint file
- `model` (BCSModel): Model to load weights into
- `optimizer` (Optional): Optimizer to restore state (for resuming training)
- `device` (str): Device to load tensors to

**Returns**: Checkpoint metadata dictionary

**Side Effects**:
- Updates `model` weights in-place
- Updates `optimizer` state if provided

---

### Function: `create_optimizer(model: BCSModel, config: Dict) -> torch.optim.AdamW`

**Description**: Creates optimizer with appropriate hyperparameters.

**Returns**: Configured AdamW optimizer with:
- Learning rate scheduling (linear warmup + decay)
- Weight decay for regularization
- Separate parameter groups (BERT vs GNN)

---

### Function: `log_metrics(metrics: Dict, step: int, log_file: str) -> None`

**Description**: Appends metrics to CSV log file.

**Metrics Schema** (CSV columns):
```
epoch,step,train_loss,mlm_loss,contrastive_loss,val_loss,learning_rate,timestamp
0,10,2.543,1.234,1.309,,2e-5,2025-12-13 10:00:00
0,20,2.398,1.156,1.242,,2e-5,2025-12-13 10:00:15
1,30,2.145,0.987,1.158,1.543,1.95e-5,2025-12-13 10:01:30
```

---

## Inference Module API (Vectorization + Mathematical Similarity Search)

**Architecture Philosophy**:
- **Vectorization**: Trained model acts as feature extractor (code+graph → fixed 768-dim vector)
- **Similarity Detection**: Happens OUTSIDE the neural network using simple mathematical operations
- **No GPU Required**: Similarity search uses numpy/scipy cosine similarity (CPU-only, <1 second for 10K vectors)

### Function: `vectorize_binary(binary_path: str, model: BCSModel, checkpoint_path: str, output_dir: str, device: str = "cuda") -> Dict[str, Any]`

**Description**: Converts a binary into a fixed-size embedding vector (feature extraction, NOT classification).

**Parameters**:
- `binary_path` (str): Path to binary executable
- `model` (BCSModel): Model for feature extraction (encoder only)
- `checkpoint_path` (str): Path to trained model checkpoint
- `output_dir` (str): Directory to save embeddings and preprocessing outputs
- `device` (str): Device for vectorization (GPU accelerates this step, but not similarity search)

**Returns**: Vectorization result dictionary:
```python
{
    "binary_hash": str,
    "binary_path": str,
    "embedding": np.ndarray,  # Shape: [768] - THIS is the final output for similarity search
    "preprocessing_time": float,
    "vectorization_time": float,  # Time for model forward pass only
    "total_time": float,
    "cfg_node_count": int,
    "sequence_length": int
}
```

**Behavior**:
1. Calls preprocessing module to extract CFG (if not already done)
2. Loads preprocessed data (graph + instruction sequence)
3. Runs model forward pass to extract embedding: `embedding = model(data)['embeddings']`
4. Converts embedding to numpy array (moves from GPU to CPU if needed)
5. Saves embedding to disk: `embeddings/{binary_hash}.npy`
6. Returns results (model's job is done - vector is ready for mathematical similarity search)

**Key Distinction**: This is NOT inference in the classification sense. The model does not "predict" similarity. It only converts code → vector.
5. Saves embedding to file
6. Returns inference results

---

### Function: `compute_similarity_rankings(query_embedding: np.ndarray, embedding_database: np.ndarray, database_metadata: pd.DataFrame, top_k: int = 10, metric: str = "cosine") -> pd.DataFrame`

**Description**: Computes similarity between query and database using MATHEMATICAL distance (no neural network).

**Parameters**:
- `query_embedding` (np.ndarray): Query embedding, shape [768]
- `embedding_database` (np.ndarray): Database of embeddings, shape [N, 768]
- `database_metadata` (pd.DataFrame): Metadata for each database entry (binary_hash, source_project, function_name, etc.)
- `top_k` (int): Number of top results to return (default: 10)
- `metric` (str): Distance metric, "cosine" (default) or "euclidean"

**Returns**: DataFrame with rankings (NO GPU NEEDED):
```
| rank | binary_hash | function_name | source_project | similarity_score | distance |
|------|-------------|---------------|----------------|------------------|----------|
| 1    | abc123...   | sort          | clamav         | 0.9543          | 0.0457   |
| 2    | def456...   | sort          | openssl        | 0.9201          | 0.0799   |
| 3    | ghi789...   | parse         | curl           | 0.7856          | 0.2144   |
```

**Implementation** (pure numpy/scipy - NO TORCH):
```python
from scipy.spatial.distance import cosine, euclidean

if metric == "cosine":
    # Cosine similarity: 1 - cosine_distance
    similarities = [1 - cosine(query_embedding, db_emb) for db_emb in embedding_database]
elif metric == "euclidean":
    distances = [euclidean(query_embedding, db_emb) for db_emb in embedding_database]
    similarities = 1 / (1 + distances)  # Convert to similarity score
```

**Performance**: < 1 second for 10,000 vectors on CPU (no GPU acceleration needed)

**Sorting**: Results sorted by descending similarity score (highest = most similar)

---

### Function: `build_embedding_database(binaries: List[str], model: BCSModel, checkpoint_path: str, output_path: str, device: str = "cuda") -> Tuple[np.ndarray, pd.DataFrame]`

**Description**: Precomputes embeddings for all binaries in reference dataset (vectorization phase).

**Parameters**:
- `binaries` (List[str]): List of binary file paths
- `model` (BCSModel): Model for feature extraction
- `checkpoint_path` (str): Path to trained checkpoint
- `output_path` (str): Where to save embedding database
- `device` (str): Device for vectorization (GPU accelerates this, but not similarity search)

**Returns**: Tuple of:
- `embeddings` (np.ndarray): Shape [N, 768] - all embeddings stacked
- `metadata` (pd.DataFrame): Metadata for each embedding (binary_hash, function_name, source_project, etc.)

**Output Format**: Saves to disk as:
- `{output_path}/embeddings.npy` - numpy array [N, 768]
- `{output_path}/metadata.csv` - CSV with binary_hash, function_name, source_project, compilation_settings, etc.
- Individual files: `{output_path}/embeddings/{binary_hash}.npy` (for incremental updates)

**Performance**: Uses batching for efficiency (batch_size=32, GPU accelerates this step)

**Usage**: Run once to vectorize entire dataset, then use for all future similarity searches (no re-vectorization needed)

---

### Function: `cluster_embeddings(embeddings: np.ndarray, metadata: pd.DataFrame, method: str = "kmeans", n_clusters: int = 10) -> pd.DataFrame`

**Description**: Clusters embeddings to discover binary families (NO NEURAL NETWORK - pure sklearn/scipy).

**Parameters**:
- `embeddings` (np.ndarray): Embedding matrix, shape [N, 768]
- `metadata` (pd.DataFrame): Metadata for each embedding
- `method` (str): Clustering algorithm, "kmeans", "dbscan", or "hierarchical"
- `n_clusters` (int): Number of clusters (for k-means/hierarchical)

**Returns**: DataFrame with cluster assignments:
```
| binary_hash | function_name | source_project | cluster_id | distance_to_centroid |
|-------------|---------------|----------------|------------|---------------------|
| abc123...   | sort          | clamav         | 0          | 0.234               |
| def456...   | sort          | openssl        | 0          | 0.198               |
| ghi789...   | parse         | curl           | 1          | 0.412               |
```

**Implementation** (sklearn, no neural network):
```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

if method == "kmeans":
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    labels = clusterer.fit_predict(embeddings)
elif method == "dbscan":
    clusterer = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
    labels = clusterer.fit_predict(embeddings)
```

**Evaluation Metrics**:
- Silhouette score (higher = better cluster separation)
- Cluster purity (if ground truth labels available)

**Use Case**: Discover binary families (e.g., all "sort" functions cluster together regardless of compiler)

---

## Testing Interface

**Unit Tests**:
- `test_save_load_checkpoint()`: Save and load checkpoint, verify state matches
- `test_training_single_step()`: Run one training step, verify loss computed
- `test_early_stopping()`: Simulate plateauing loss, verify early stop triggers
- `test_inference_pipeline()`: Run inference on test binary, verify embedding shape
- `test_similarity_ranking()`: Verify rankings are sorted correctly

**Integration Test**:
- Train model on small dataset (100 samples, 2 epochs)
- Save checkpoint
- Run inference on new binary
- Compute similarity rankings
- Verify top result is from correct project

---

## Example Usage

### Training

```python
from pipeline.train import train_model, load_checkpoint
from pipeline.dataset import BCSDataset, collate_fn
from torch.utils.data import DataLoader

# Load datasets
train_dataset = BCSDataset("data", split="train")
val_dataset = BCSDataset("data", split="validation")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)

# Create model
model = BCSModel(gnn_config, bert_config).to("cuda")

# Training configuration
config = {
    "epochs": 10,
    "learning_rate": 2e-5,
    "lambda_contrastive": 0.5,
    "checkpoint_dir": "checkpoints/",
    "log_dir": "logs/",
    "seed": 42
}

# Train
results = train_model(config, train_loader, val_loader, model, device="cuda")
print(f"Best checkpoint: {results['best_checkpoint_path']}")
print(f"Best val loss: {results['best_val_loss']:.4f}")
```

### Inference

```python
from pipeline.inference import run_inference, compute_similarity_rankings, build_embedding_database

# Build database from training set
train_binaries = [...]  # List of training binary paths
embedding_db = build_embedding_database(
    train_binaries,
    model,
    checkpoint_path="checkpoints/model_epoch_10_valloss_0.2345.pt",
    device="cuda"
)

# Run inference on test binary
test_result = run_inference(
    binary_path="/path/to/test/binary",
    model=model,
    checkpoint_path="checkpoints/model_epoch_10_valloss_0.2345.pt",
    output_dir="data/preprocessed",
    device="cuda"
)

# Get similarity rankings
rankings = compute_similarity_rankings(
    test_binary_hash=test_result["binary_hash"],
    embedding_database=embedding_db,
    top_k=10
)

print(rankings)
```

---

## Performance Specifications

| Operation | Expected Time | Notes |
|-----------|---------------|-------|
| Single training step (batch=16) | <300 ms | Includes forward, backward, optimizer step |
| Full epoch (1000 samples) | <5 minutes | On single GPU |
| Checkpoint save | <1 second | ~500 MB file |
| Single inference (with preprocessing) | <5 minutes | Dominated by angr preprocessing |
| Single inference (pre-preprocessed) | <100 ms | Just model forward pass |
| Build embedding database (1000 binaries) | <10 minutes | Using batch inference |

---

## Error Handling

**Training Errors**:
- OOM (Out of Memory) → Suggest reducing batch_size in config
- Diverging loss (NaN) → Stop training, log hyperparameters, suggest reducing learning_rate
- Checkpoint save failure → Log error, continue training (don't crash)

**Inference Errors**:
- Preprocessing failure → Return error status, don't crash pipeline
- Model load failure → Raise `FileNotFoundError` with clear message
- Shape mismatch → Raise `ValueError` with expected vs actual shapes

---

## Dependencies

**Required Libraries**:
- torch >= 2.0
- transformers >= 4.30
- pandas >= 1.5
- numpy >= 1.24
- tqdm (for progress bars)
- gitpython (for logging git commit hash)
