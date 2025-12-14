# Model Module Contract (PHASE 3)

**Modules**: `models.gnn_encoder` + `models.custom_attention` + `models.bert_encoder` + `models.bcsd_model`  
**Purpose**: Graph encoding, custom KV-Prefix attention (HARD PART), BERT integration, main BCSD model  
**Owner**: User Stories 3, 4, 5 (P3-P5)

---

## Module Overview

**PHASE 3** contains the neural architecture with four key components:

1. **GNN Encoder** (`models/gnn_encoder.py`):
   - GAT layers for graph summarization
   - Produces fixed-size Graph Summary vector from variable-size CFGs

2. **Custom Attention** (`models/custom_attention.py`):
   - **THIS IS THE HARD PART** - Novel research contribution
   - KV-Prefix attention mechanism
   - Injects graph-derived Keys and Values into BERT's attention

3. **BERT Encoder** (`models/bert_encoder.py`):
   - BERT model modified to use custom attention
   - Integrates graph structure with sequential information

4. **BCSD Model** (`models/bcsd_model.py`):
   - "God Class" orchestrating GNN→Projection→BERT pipeline
   - Main entry point for training and inference

---

## GNN Module API

### Class: `GATEncoder(torch.nn.Module)`

**Description**: Graph Attention Network encoder for CFG summarization.

**Constructor**: `__init__(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3, heads: int = 4, dropout: float = 0.2)`

**Parameters**:
- `input_dim` (int): Dimension of input node features
- `hidden_dim` (int): Hidden dimension for GAT layers
- `output_dim` (int): Output graph summary dimension (default: 256)
- `num_layers` (int): Number of GAT layers (default: 3)
- `heads` (int): Number of attention heads per layer (default: 4)
- `dropout` (float): Dropout rate (default: 0.2)

**Methods**:

#### `forward(x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor`

**Parameters**:
- `x` (Tensor): Node features, shape [num_nodes, input_dim]
- `edge_index` (Tensor): Graph connectivity, shape [2, num_edges]
- `batch` (Tensor): Batch assignment vector, shape [num_nodes]

**Returns**: Graph summaries, shape [batch_size, output_dim]

**Behavior**:
- Applies 3 GAT layers with multi-head attention
- Uses LeakyReLU activation between layers
- Applies dropout after each layer
- Performs attention-based global pooling to produce fixed-size output

---

### Function: `initialize_node_features(instructions: List[List[str]], tokenizer, embedding_layer) -> Tensor`

**Description**: Initializes node features from instruction sequences.

**Parameters**:
- `instructions` (List[List[str]]): Instructions for each basic block
- `tokenizer`: BERT tokenizer
- `embedding_layer`: BERT embedding layer

**Returns**: Node feature matrix, shape [num_nodes, 768]

**Strategy**:
- Tokenize instructions in each basic block
- Get BERT embeddings for tokens
- Average embeddings to get single vector per basic block (node)

---

## BERT Integration Module API

### Class: `BERTWithGraphPrefix(torch.nn.Module)`

**Description**: BERT model with custom KV-Prefix attention for graph integration. Implements **Deep Prefix Injection** where graph summary is injected into EVERY Transformer layer.

**Architecture Note**: 
- Graph summary acts as "global memory" visible at all depths
- Each BERT layer receives the same graph_summary (but different projections per layer)
- Alternative approach: Could recompute prefix per layer, but shared prefix is simpler and effective

**Constructor**: `__init__(bert_model_name: str = "bert-base-uncased", graph_dim: int = 256, freeze_embeddings: bool = True)`

**Parameters**:
- `bert_model_name` (str): Pretrained BERT model identifier
- `graph_dim` (int): Dimension of graph summary input
- `freeze_embeddings` (bool): Whether to freeze BERT embeddings initially

**Methods**:

#### `forward(input_ids: Tensor, attention_mask: Tensor, graph_summary: Tensor) -> Dict[str, Tensor]`

**Parameters**:
- `input_ids` (Tensor): Tokenized input, shape [batch_size, seq_len]
- `attention_mask` (Tensor): Attention mask, shape [batch_size, seq_len]
- `graph_summary` (Tensor): Graph summaries from GNN, shape [batch_size, graph_dim]

**Returns**: Dictionary with:
```python
{
    "cls_embedding": Tensor,  # [batch_size, 768]
    "sequence_output": Tensor,  # [batch_size, seq_len, 768]
    "attention_weights": Tensor,  # [batch_size, num_heads, seq_len, seq_len+1]
    "pooled_output": Tensor  # [batch_size, 768]
}
```

**Visual Diagram**:
```
Input: Assembly Tokens + CFG Graph
         ↓                    ↓
    [CLS] T1 T2 T3        [GNN Encoder]
         ↓                    ↓
  Token Embeddings      Graph Summary (256-dim)
         ↓                    ↓
         ├────────────────────┤
         ↓                    
    LAYER 1 (Transformer with KV-Prefix):
         graph_summary → [Linear_K, Linear_V] → [prefix_k, prefix_v]
         token Q, K, V + concatenate([prefix_k, prefix_v], [K, V])
         → K_total: [Prefix | T_CLS | T1 | T2 | T3]
         → V_total: [Prefix | T_CLS | T1 | T2 | T3]
         → Attention(Q_tokens, K_total, V_total)
         ↓
    LAYER 2 (Same graph_summary, different projections):
         graph_summary → [Linear_K, Linear_V] → [prefix_k, prefix_v]
         ... (repeat concatenation + attention)
         ↓
    ... (Layers 3-12, all with prefix injection)
         ↓
    Final Layer Output
         ↓
    Extract [CLS] → Final Embedding (768-dim)
```

**Key Properties**:
- Sequence length at each layer: seq_len + 1 (due to prefix)
- Each layer has independent graph_to_k and graph_to_v projections
- Prefix position is always first in concatenated K/V (position 0)
- All tokens attend to prefix at every layer

**Behavior** (Deep Prefix Injection):
1. Get BERT token embeddings from input_ids
2. For EACH of the 12 Transformer layers:
   - Pass graph_summary to that layer's KVPrefixAttention module
   - Layer projects graph_summary → prefix_k, prefix_v
   - Layer concatenates prefix to sequence K/V (extends length by 1)
   - Standard attention computed with extended K/V
3. Extract [CLS] token embedding from final layer
4. Returns both token-level and sentence-level embeddings

**Key Insight**: Same graph_summary flows through all 12 layers, but each layer has independent projection weights (graph_to_k, graph_to_v). This allows each layer to extract different aspects of the graph structure.

**Integration with Hugging Face BERT**:

Option 1: **Monkey-Patch Attention Layers** (Simpler)
```python
from transformers import BertModel

bert = BertModel.from_pretrained("bert-base-uncased")

# Replace attention in each layer
for layer in bert.encoder.layer:
    original_attention = layer.attention.self
    layer.attention.self = KVPrefixAttentionWrapper(
        original_attention, 
        graph_dim=256
    )
```

Option 2: **Custom BERT Implementation** (More Control)
```python
# Copy BertLayer code and modify the attention forward pass
class CustomBertLayer(nn.Module):
    def __init__(self, config, graph_dim=256):
        super().__init__()
        self.attention = CustomBertAttention(config, graph_dim)
        # ... rest of layer (FFN, LayerNorm, etc.)
    
    def forward(self, hidden_states, attention_mask, graph_summary):
        # Pass graph_summary to attention
        attention_output = self.attention(
            hidden_states, attention_mask, graph_summary
        )
        # ... rest of forward pass
```

**Recommendation**: Start with Option 1 for prototyping, move to Option 2 for production if needed.

---

## API: Custom Attention Module (THE HARD PART)

**Module**: `models.custom_attention`

### Class: `KVPrefixAttention(torch.nn.Module)`

**Description**: Custom attention layer that injects graph-derived K/V prefix into BERT attention. **This is the novel research contribution and the most complex component.**

**Research Context**: Standard BERT attention operates on sequence tokens only. This mechanism extends attention to include a "prefix" derived from the graph structure, allowing the model to attend to both sequential (instructions) and structural (CFG) information simultaneously.

**Constructor**: `__init__(hidden_size: int = 768, num_heads: int = 12, graph_dim: int = 256)`

**Parameters**:
- `hidden_size` (int): BERT hidden dimension (768 for bert-base)
- `num_heads` (int): Number of attention heads (12 for bert-base)
- `graph_dim` (int): Dimension of graph summary from GNN

**Internal Components**:
- `graph_to_k`: Linear projection from graph_dim to hidden_size (separate projection for Keys)
- `graph_to_v`: Linear projection from graph_dim to hidden_size (separate projection for Values)
- `split_heads`: Function to reshape [batch, hidden] to [batch, heads, 1, head_dim]
- `merge_heads`: Function to reshape back after attention computation

**Why Separate K/V Projections**: Using two separate linear layers (`graph_to_k` and `graph_to_v`) instead of one combined projection gives the model more flexibility to learn different transformations for Keys vs Values from the graph structure.

**Methods**:

#### `forward(query: Tensor, key: Tensor, value: Tensor, graph_summary: Tensor, attention_mask: Tensor) -> Tuple[Tensor, Tensor]`

**Parameters**:
- `query` (Tensor): Query from BERT tokens, shape [batch, heads, seq_len, head_dim]
- `key` (Tensor): Key from BERT tokens, shape [batch, heads, seq_len, head_dim]
- `value` (Tensor): Value from BERT tokens, shape [batch, heads, seq_len, head_dim]
- `graph_summary` (Tensor): Graph summary from GNN, shape [batch, graph_dim]
- `attention_mask` (Tensor): Mask for padding, shape [batch, seq_len]

**Returns**: 
- `context` (Tensor): Attended output, shape [batch, heads, seq_len, head_dim]
- `attention_weights` (Tensor): Attention scores, shape [batch, heads, seq_len, seq_len+1]

**Algorithm** (THE HARD PART - Concatenation NOT Addition):
```python
# CRITICAL: This extends sequence length, does NOT modify token vectors directly

# Step 1: Project graph summary to separate K and V prefixes
prefix_k = self.graph_to_k(graph_summary)  # [batch, hidden_dim]
prefix_v = self.graph_to_v(graph_summary)  # [batch, hidden_dim]

# Step 2: Reshape for multi-head attention
prefix_k = self.split_heads(prefix_k)  # [batch, heads, 1, head_dim]
prefix_v = self.split_heads(prefix_v)  # [batch, heads, 1, head_dim]

# Step 3: CONCATENATE (NOT ADD) prefixes along LENGTH dimension
# This extends the sequence: seq_len → seq_len+1
K_total = torch.cat([prefix_k, key], dim=2)    # [batch, heads, 1+seq_len, head_dim]
V_total = torch.cat([prefix_v, value], dim=2)  # [batch, heads, 1+seq_len, head_dim]
# query remains unchanged: [batch, heads, seq_len, head_dim]

# Step 4: Extend attention mask (prefix position always visible)
extended_mask = torch.cat([
    torch.ones(batch, 1, device=attention_mask.device),  # Prefix always attended
    attention_mask  # Original sequence mask
], dim=1)  # [batch, 1+seq_len]

# Step 5: Standard scaled dot-product attention with extended K/V
scores = torch.matmul(query, K_total.transpose(-2, -1))  # [batch, heads, seq_len, 1+seq_len]
scores = scores / math.sqrt(head_dim)
scores = scores.masked_fill(extended_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
attn_weights = F.softmax(scores, dim=-1)
context = torch.matmul(attn_weights, V_total)  # [batch, heads, seq_len, head_dim]

return context, attn_weights
```

**Critical Implementation Notes**:
1. **Concatenation NOT Addition**: Graph prefix EXTENDS the sequence (seq_len → seq_len+1), does NOT modify existing token vectors
2. **Deep Prefix Injection**: Graph prefix must be injected into EVERY Transformer layer, not just the first one (ensures CFG structure remains visible as "global memory" at all depths)
3. Graph prefix is ALWAYS attended to (mask value = 1)
4. Position 0 in attention weights corresponds to graph summary prefix
5. Positions 1+ correspond to sequence tokens
6. Gradients must flow back to graph_summary projection
7. Head dimension must match: head_dim = hidden_size / num_heads
8. Use separate linear projections for K and V (graph_to_k, graph_to_v) - more flexible than single projection

**Testing Strategy**:
- Unit test with dummy tensors (batch=2, seq_len=10, graph_dim=256)
- Verify output shapes match expectations
- Verify gradients flow to all inputs
- Visualize attention weights (should show tokens attending to graph)

**Implementation Skeleton** (`models/custom_attention.py`):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KVPrefixAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, graph_dim=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Separate projections for K and V from graph
        self.graph_to_k = nn.Linear(graph_dim, hidden_size)
        self.graph_to_v = nn.Linear(graph_dim, hidden_size)
        
    def split_heads(self, x):
        """Reshape [batch, hidden] → [batch, heads, 1, head_dim]"""
        batch_size = x.shape[0]
        x = x.view(batch_size, self.num_heads, 1, self.head_dim)
        return x
    
    def forward(self, query, key, value, graph_summary, attention_mask):
        """
        query, key, value: [batch, heads, seq_len, head_dim]
        graph_summary: [batch, graph_dim]
        attention_mask: [batch, seq_len]
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Project graph to K and V prefixes
        prefix_k = self.graph_to_k(graph_summary)  # [batch, hidden]
        prefix_v = self.graph_to_v(graph_summary)  # [batch, hidden]
        
        # Reshape for multi-head
        prefix_k = self.split_heads(prefix_k)  # [batch, heads, 1, head_dim]
        prefix_v = self.split_heads(prefix_v)  # [batch, heads, 1, head_dim]
        
        # CONCATENATE along length dimension
        K_total = torch.cat([prefix_k, key], dim=2)  # [batch, heads, 1+seq_len, head_dim]
        V_total = torch.cat([prefix_v, value], dim=2)
        
        # Extend attention mask
        extended_mask = torch.cat([
            torch.ones(batch_size, 1, device=attention_mask.device),
            attention_mask
        ], dim=1)  # [batch, 1+seq_len]
        
        # Scaled dot-product attention
        scores = torch.matmul(query, K_total.transpose(-2, -1))  # [batch, heads, seq_len, 1+seq_len]
        scores = scores / math.sqrt(head_dim)
        
        # Apply mask (reshape for broadcasting)
        mask_reshaped = extended_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, 1+seq_len]
        scores = scores.masked_fill(mask_reshaped == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V_total)  # [batch, heads, seq_len, head_dim]
        
        return context, attn_weights
```

---

## Joint Model API

### Class: `BCSModel(torch.nn.Module)`

**Description**: Complete model integrating GNN encoder and BERT with custom attention. This is a **single-tower architecture** (one model instance with one input interface), NOT a dual-branch model.

**Architecture Note**:
- **Single Tower**: One unified model with shared weights for all inputs
- **Siamese Training**: The same model instance processes multiple samples sequentially within a batch
- **Not Dual-Branch**: Does NOT have separate encoders for "input_A" and "input_B"
- **Feature Extractor**: Primary purpose is to convert code+graph → fixed 768-dim embedding vector

**Constructor**: `__init__(gnn_config: Dict, bert_config: Dict)`

**Parameters**:
- `gnn_config` (Dict): Configuration for GATEncoder
- `bert_config` (Dict): Configuration for BERTWithGraphPrefix

**Methods**:

#### `forward(batch: Dict) -> Dict[str, Tensor]`

**Parameters**:
- `batch` (Dict): Output from DataLoader (see dataset_module.md) containing ONE sample's data

**Returns**: Dictionary with:
```python
{
    "embeddings": Tensor,  # [batch_size, 768] - for similarity computation
    "mlm_logits": Tensor,  # [batch_size, seq_len, vocab_size] - for MLM loss
    "graph_summary": Tensor,  # [batch_size, 256] - intermediate output
    "attention_weights": Tensor  # for visualization
}
```

**Processing Note**: 
- Forward pass processes each sample individually with the same weights
- For Siamese training, call `forward()` separately for each member of a pair:
  ```python
  output_A = model(sample_A)  # Uses model weights
  output_B = model(sample_B)  # Uses SAME model weights
  loss = contrastive_loss(output_A['embeddings'], output_B['embeddings'])
  ```

#### `compute_similarity(embedding1: Tensor, embedding2: Tensor) -> Tensor`

**Parameters**:
- `embedding1` (Tensor): First embedding, shape [batch_size, 768]
- `embedding2` (Tensor): Second embedding, shape [batch_size, 768]

**Returns**: Cosine similarity scores, shape [batch_size]

**Formula**: `cosine_sim = (emb1 · emb2) / (||emb1|| * ||emb2||)`

**Usage Context**: This is a utility function for loss computation during training. Inference uses numpy/scipy for similarity search (no GPU required).

---

## Training Loop Interface

### Function: `train_epoch(model: BCSModel, dataloader: DataLoader, optimizer, loss_fn: Dict, device: str) -> Dict[str, float]`

**Description**: Trains model for one epoch.

**Parameters**:
- `model` (BCSModel): Model to train
- `dataloader` (DataLoader): Training data
- `optimizer`: PyTorch optimizer (AdamW recommended)
- `loss_fn` (Dict): Dictionary with "mlm" and "contrastive" loss functions
- `device` (str): "cuda" or "cpu"

**Returns**: Dictionary with average losses:
```python
{
    "train_loss": float,
    "mlm_loss": float,
    "contrastive_loss": float
}
```

---

### Function: `validate(model: BCSModel, dataloader: DataLoader, loss_fn: Dict, device: str) -> Dict[str, float]`

**Description**: Validates model on validation set.

**Parameters**: Same as `train_epoch`

**Returns**: Validation losses (same structure as `train_epoch`)

---

## Loss Functions

### MLM Loss

**Function**: `masked_language_modeling_loss(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor`

**Description**: Cross-entropy loss for masked token prediction.

**Parameters**:
- `logits` (Tensor): Model predictions, shape [batch, seq_len, vocab_size]
- `labels` (Tensor): Ground truth token IDs, shape [batch, seq_len]
- `mask` (Tensor): Binary mask indicating which tokens are masked, shape [batch, seq_len]

**Returns**: Scalar loss value

**Formula**: `loss = CrossEntropy(logits[mask], labels[mask])`

---

### Contrastive Loss

**Function**: `nt_xent_loss(embeddings: Tensor, labels: Tensor, temperature: float = 0.07) -> Tensor`

**Description**: NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning.

**Parameters**:
- `embeddings` (Tensor): Batch of embeddings, shape [batch_size, 768]
- `labels` (Tensor): Similarity labels (1 = similar, 0 = dissimilar), shape [batch_size]
- `temperature` (float): Temperature scaling parameter (default: 0.07)

**Returns**: Scalar loss value

**Formula**:
```
sim_matrix = embeddings @ embeddings.T / temperature
loss = -log(exp(sim_matrix[i,j]) / sum(exp(sim_matrix[i,k])))
       for positive pairs (i,j)
```

---

## Testing Interface

**Unit Tests**:
- `test_gnn_forward()`: Pass dummy graph through GNN, verify output shape [batch, 256]
- `test_gnn_varying_sizes()`: Test graphs with different node counts
- `test_kv_prefix_dimensions()`: Verify attention output has correct shape
- `test_bert_with_graph()`: Forward pass with graph prefix, check output
- `test_similarity_computation()`: Verify cosine similarity in [-1, 1]
- `test_gradient_flow()`: Backward pass, check gradients exist for all parameters

**Integration Test**:
- Create dummy batch (2 samples, 50 tokens, 20 nodes)
- Forward pass through BCSModel
- Compute losses (MLM + contrastive)
- Backward pass and optimizer step
- Verify loss decreases over 10 iterations

---

## Example Usage

```python
from pipeline.gnn_model import GATEncoder, BCSModel
from pipeline.bert_encoder import BERTWithGraphPrefix

# Initialize model
model = BCSModel(
    gnn_config={
        "input_dim": 768,
        "hidden_dim": 256,
        "output_dim": 256,
        "num_layers": 3,
        "heads": 4
    },
    bert_config={
        "bert_model_name": "bert-base-uncased",
        "graph_dim": 256,
        "freeze_embeddings": True
    }
).to("cuda")

# Forward pass
batch = next(iter(train_loader))
outputs = model(batch)

embeddings = outputs["embeddings"]  # [16, 768]
mlm_logits = outputs["mlm_logits"]  # [16, seq_len, vocab_size]

# Compute loss
mlm_loss = masked_language_modeling_loss(mlm_logits, batch["labels"], batch["mlm_mask"])
contrastive_loss = nt_xent_loss(embeddings, batch["pair_labels"])

total_loss = mlm_loss + 0.5 * contrastive_loss
total_loss.backward()
optimizer.step()
```

---

## Performance Specifications

| Operation | Expected Time (GPU) | Memory |
|-----------|---------------------|--------|
| GNN forward (batch=16, 100 nodes) | <10 ms | <200 MB |
| BERT forward (batch=16, seq_len=512) | <50 ms | <1 GB |
| Combined forward pass | <100 ms | <2 GB |
| Backward pass | <150 ms | <4 GB |
| Full training step | <250 ms | <4 GB |

---

## Dependencies

**Required Libraries**:
- torch >= 2.0
- torch_geometric >= 2.3
- transformers >= 4.30 (Hugging Face)
- numpy >= 1.24
