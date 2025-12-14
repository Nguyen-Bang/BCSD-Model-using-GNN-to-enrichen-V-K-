# Research & Technical Decisions

**Feature**: Complete BCSD Pipeline Implementation  
**Date**: 2025-12-13  
**Status**: Phase 0 - Research Complete

## Overview

This document captures technical research, design decisions, and rationale for implementation choices in the BCSD pipeline. All decisions are made to support the central hypothesis: integrating CFG structure via GNN with BERT embeddings improves binary similarity detection.

---

## 1. Binary Analysis Framework Selection

### Decision: angr with CFGFast

**Test-Driven Development Strategy**:
- **Test Binary**: `test_binaries/test_gnn.c` - Simple C program with three functions demonstrating key CFG patterns:
  - `calculate_sum()`: Contains loop (creates backward edge/cycle in CFG)
  - `check_value()`: Contains if/else branches (creates multiple paths in CFG)
  - `main()`: Contains function calls and conditional execution
- **Purpose**: Validate disassembly correctness and tokenization strategy before processing large dataset
- **Thesis Documentation**: Use test binary as concrete example to illustrate angr workflow, CFG structure, and instruction tokenization in methodology chapter
- **Validation Workflow**:
  1. Compile test_gnn.c with multiple compilation variants (gcc/clang, O0/O3)
  2. Run angr CFGFast on each variant
  3. Manually inspect extracted CFG JSON to verify:
     - Loop detection: Backward edge from loop body to condition check in calculate_sum
     - Branch detection: Multiple outgoing edges from conditional blocks in check_value
     - Function boundaries: Correct identification of three separate functions
     - Instruction tokenization: Opcodes and operands correctly separated
  4. Compare CFG structures across compilation variants to observe optimization effects
  5. Use validated output as running example in thesis methodology chapter

**Rationale**:
- angr is the industry-standard framework for binary analysis in Python
- CFGFast provides accurate CFG extraction without full symbolic execution overhead
- Built-in support for multiple architectures (x86, ARM, MIPS)
- Active community and extensive documentation
- Integrates well with Python ML ecosystem

**Alternatives Considered**:
- **Ghidra + PyGhidra**: More accurate decompilation but heavier weight, harder to automate
- **radare2 + r2pipe**: Lighter weight but less structured CFG output, steeper learning curve
- **IDA Pro + IDAPython**: Commercial, expensive, not suitable for open research

**Implementation Notes**:
- Use `CFGFast(auto_load_libs=False)` to avoid analyzing library code
- Handle stripped binaries gracefully by using address-based node identification
- Extract basic block boundaries and instruction sequences separately
- Serialize CFG as edge list + node metadata (JSON format for portability)

**Best Practices**:
- Run preprocessing offline to avoid blocking training loop
- Implement timeout mechanisms for large binaries (>10MB)
- Cache CFG results to avoid re-analysis
- Log warnings for failed analyses but don't crash pipeline

---

## 2. Custom Assembly Tokenization Strategy

### Decision: Domain-Specific Tokenizer (NOT BERT's WordPiece)

**Problem**: BERT's WordPiece tokenizer is designed for natural language:
- Splits unknown words into subword units (e.g., "movq" → "mov", "##q")
- Designed for English vocabulary, not assembly language
- Cannot handle hex addresses, register names, immediate values effectively
- Creates unnecessarily large token sequences

**Solution**: Build custom tokenizer specifically for assembly instructions

**Tokenization Strategy**:
1. **Instruction Parsing**: 
   - Split each instruction: `"mov rax, rdi"` → `["mov", "rax", "rdi"]`
   - Handle common patterns: `"call 0x401000"` → `["call", "0x401000"]`
   
2. **Vocabulary Building**:
   - Collect all unique opcodes (mov, add, jmp, call, etc.) → ~200 opcodes for x86_64
   - Collect all unique registers (rax, rbx, eax, al, etc.) → ~100 registers
   - Collect immediate values and addresses (keep as-is or group by type)
   - Total vocab size: up to 5000 tokens (configurable, much smaller than BERT's 30K)
   - If training data contains <5000 unique tokens, actual vocab will be smaller

3. **Special Tokens**:
   - `[PAD]` = 0 (padding token)
   - `[CLS]` = 101 (classification token, start of sequence)
   - `[SEP]` = 102 (separator token, end of sequence)
   - `[MASK]` = 103 (masking token for MLM training)
   - `[UNK]` = 104 (unknown token for rare values)

4. **Token ID Assignment**:
   - Sort tokens by frequency (most common opcodes get lower IDs)
   - Ensures common patterns have consistent representations
   - Preserves domain semantics (opcodes grouped, registers grouped)

**Rationale**:
- **Domain Specificity**: Assembly has fixed vocabulary (opcodes are finite)
- **Efficiency**: Smaller vocab → faster training, less memory
- **Interpretability**: Token IDs directly correspond to assembly concepts
- **Control**: Can enforce semantic groupings (all jump instructions similar IDs)

**Implementation** (`preprocessing/tokenizer.py`):
```python
class AssemblyTokenizer:
    def __init__(self, vocab_size=5000, max_seq_length=512):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.vocab = {}  # token -> id
        self.reverse_vocab = {}  # id -> token
        
    def build_vocab(self, disasm_files):
        # Collect all tokens from disassembly files
        # Build frequency-based vocab
        # Assign IDs
        pass
    
    def tokenize(self, instructions):
        # Split instructions into tokens
        # Map tokens to IDs
        # Add [CLS], [SEP], padding
        # Generate attention mask
        return {
            'token_ids': [...],
            'attention_mask': [...]
        }
```

**Alternatives Considered**:
- **Use BERT Tokenizer**: Simple but loses domain structure, creates subword fragmentation
- **Character-Level**: Preserves everything but loses semantic groupings, longer sequences
- **Byte-Pair Encoding (BPE)**: Middle ground but still treats assembly as text, not structured data

**Best Practices**:
- Build vocabulary from training data only (avoid data leakage from test set)
- Handle rare tokens with [UNK] (don't let vocab explode)
- Save vocabulary to JSON file for reproducibility
- Validate tokenization on test_binaries before full dataset

**Integration with BERT**:
- Custom tokenizer produces token IDs
- BERT's embedding layer maps IDs to vectors (will be trained from scratch or initialized randomly)
- Rest of BERT architecture unchanged (same attention, FFN, etc.)
- Critical: Do NOT use pretrained BERT embeddings (they expect WordPiece token IDs)

---

## 3. Graph Neural Network Architecture

### Decision: Graph Attention Networks (GAT) with PyTorch Geometric

**Rationale**:
- GAT learns importance of different neighbors via attention mechanism
- More expressive than standard GCN for control flow graphs with varying edge semantics
- PyTorch Geometric provides optimized implementation with batching support
- Attention weights can be visualized for interpretability (thesis requirement)
- Handles graphs of varying sizes naturally

**Alternatives Considered**:
- **Graph Convolutional Networks (GCN)**: Simpler but treats all neighbors equally, less suitable for CFG where edge types matter (conditional vs unconditional jumps)
- **GraphSAGE**: Good for large graphs but less interpretable, doesn't leverage edge semantics
- **Message Passing Neural Networks (MPNN)**: More general but requires more implementation effort

**Architecture Details**:
- **Input**: Node features initialized from instruction embeddings (average of instruction token embeddings from BERT tokenizer)
- **Hidden layers**: 3 GAT layers with 256 hidden units each
- **Attention heads**: 4 heads per layer (multi-head attention for robustness)
- **Activation**: LeakyReLU (negative_slope=0.2) after each layer
- **Dropout**: 0.2 after each layer to prevent overfitting
- **Global pooling**: Attention-based pooling (learn importance of nodes for final summary)
- **Output**: 256-dimensional graph summary vector

**Best Practices**:
- Use batch processing with PyTorch Geometric's Batch class
- Normalize node features before GNN layers (mean=0, std=1)
- Implement gradient clipping (max_norm=1.0) to prevent exploding gradients
- Monitor attention weights during training for sanity checks

---

## 3. BERT Model Selection and Integration

### Decision: bert-base-uncased with Custom Attention Layer Modification

**Note**: Detailed implementation algorithm and code skeleton are in `contracts/model_module.md`. This section provides research rationale and high-level design.

**Rationale**:
- `bert-base-uncased` is well-documented, widely used, and fits in 6GB VRAM
- 768-dimensional embeddings provide rich semantic space
- Pretrained weights transfer well to code-like sequences (prior work on CodeBERT)
- Uncased version suitable since assembly instructions are case-insensitive

**Alternatives Considered**:
- **CodeBERT**: Specialized for code but trained on source code, not assembly
- **bert-large**: Better performance but requires 12GB+ VRAM, exceeds constraints
- **DistilBERT**: Faster but reduced capacity may hurt performance on complex binaries
- **RoBERTa**: Similar to BERT but different tokenization, less documentation for modification

**KV-Prefix Attention Mechanism** (Concatenation Strategy):

**Concept**: Inject graph-derived information by EXTENDING the sequence length, NOT by modifying token vectors directly. Graph summary acts as a "prefix token" that all sequence tokens can attend to.

**Critical Distinction**: 
- ❌ **NOT Addition**: We do NOT add graph features to token embeddings
- ✅ **Concatenation**: We concatenate graph-derived K/V to sequence K/V along length dimension
- **Result**: Sequence length increases from seq_len to seq_len+1

**Implementation Approach**:
1. Extract graph summary from GNN (256-dim vector)
2. Project graph summary separately into Keys and Values:
   - `prefix_k = Linear_K(graph_summary)` → 768-dim
   - `prefix_v = Linear_V(graph_summary)` → 768-dim
   - Use separate projections for flexibility (K and V learn different transformations)
3. Reshape prefixes for multi-head structure:
   - [batch_size, 768] → [batch_size, num_heads=12, 1, head_dim=64]
4. **Deep Prefix Injection**: In EVERY BERT self-attention layer (all 12 layers):
   - Concatenate prefix_k to text keys: `K_total = cat([prefix_k, K_text], dim=length)`
   - Concatenate prefix_v to text values: `V_total = cat([prefix_v, V_text], dim=length)`
   - Query remains unchanged (from text tokens only)
   - Original: Q, K, V all have shape [batch, heads, seq_len, head_dim]
   - Modified: K_total and V_total have shape [batch, heads, seq_len+1, head_dim]
   - Q remains [batch, heads, seq_len, head_dim]
5. Compute standard scaled dot-product attention with extended K/V:
   - Each text token can attend to graph prefix (position 0) + all other text tokens (positions 1+)
6. **Why Every Layer**: Ensures CFG structure remains visible as "global memory" even at deepest layers

**Why This Works**:
- Graph information is available to every token via attention mechanism (not just [CLS])
- Gradients flow back to GNN during training (end-to-end optimization)
- Minimal architectural change to BERT (only extends K/V in attention computation)
- Interpretable: can visualize how much each token attends to graph prefix
- Deep injection ensures CFG constraints visible at all layers (not just shallow)

**Alternatives Considered**:
- **Addition to Embeddings**: Add graph vector to all token embeddings → corrupts token semantics, harder to learn
- **Single-Layer Injection**: Inject prefix only in first layer → graph info dilutes through layers, not visible at depth
- **Simple [CLS] Concatenation**: Concatenate graph vector to [CLS] token embedding → loses fine-grained per-token interaction
- **Cross-Attention**: Separate cross-attention layer → more parameters, slower, risk of overfitting
- **Graph-as-Tokens**: Convert graph to token sequence → loses structural information, sequence length explosion

**Best Practices**:
- Freeze BERT embeddings initially, fine-tune attention layers only (fewer parameters)
- After initial training, unfreeze all layers for full fine-tuning
- Use gradient checkpointing to reduce memory usage during backward pass
- Monitor attention entropy (should be moderate, not too uniform or too peaked)

---

## 4. Training Strategy and Loss Functions

### Decision: Siamese Network Training with Joint MLM + Contrastive Loss

**Siamese Network Architecture Clarification**:

A Siamese network is **NOT** a dual-branch model with separate encoders. It is a **training strategy** where:
- **Single Tower**: One model instance (GNN + BERT) with one input interface
- **Shared Weights**: The same parameters process all inputs in a batch
- **Sequential Processing**: The model processes func_A, then func_B, then func_C, etc., all using identical weights
- **Contrastive Learning**: Loss is computed by comparing the resulting embeddings

**Key Distinction**:
- ❌ NOT "two-branch model" with separate encoders for each input
- ✅ A single model that acts as a **feature extractor**, processing inputs one at a time
- The "Siamese" property emerges from the training procedure (comparing pairs), not from model architecture duplication

**Two-Stage Training Pipeline**:

**Stage 1: Pretraining with MLM (5 epochs)**
- Mask 15% of instruction tokens randomly
- Train model to predict masked tokens using both sequence context and graph structure
- Input: Individual functions (no pairs needed at this stage)
- Goal: Learn semantic relationships between assembly instructions and CFG structure
- Data: All training binaries (clamav, curl, nmap, openssl)

**Stage 2: Contrastive Fine-tuning with Positive Pairs (5 epochs)**
- **Positive Pair Construction**:
  - Take the same function compiled with different settings
  - Example: `sort_gcc_O0` and `sort_clang_O3` → both implement the same `sort` function
  - Source: BinaryCorp dataset provides function-level matching across compilation variants
  - NOT just "binaries from the same project" but **semantically identical functions**
  
- **In-Batch Negatives**:
  - All other samples in the batch that are not the positive pair for a given anchor
  - For anchor `sort_gcc_O0`, positive is `sort_clang_O3`, negatives are all other functions in batch (e.g., `search_gcc_O0`, `parse_clang_O3`, etc.)
  - Efficient: No pre-mining required, automatically generated from batch composition
  
- **Siamese Processing**:
  - Model processes each function individually: forward(func_1) → emb_1, forward(func_2) → emb_2, etc.
  - All functions processed with the same model weights (no separate branches)
  - Embeddings extracted (CLS token or pooled output, 768-dim vectors)
  - Loss computed by comparing embeddings using cosine similarity

**Joint Loss Function**:
```
L_total = L_MLM + λ * L_contrastive
```
Where λ is tuned based on validation loss (initial value: 0.5)

**Contrastive Loss Implementation (InfoNCE/NT-Xent)**:
```python
# For each positive pair (anchor, positive) in batch:
# Compute similarity between anchor and positive, and anchor and all negatives

loss = -log( exp(sim(anchor, positive) / τ) / Σ exp(sim(anchor, all_in_batch) / τ) )
```
- Temperature parameter τ = 0.07 (controls separation between positive/negative pairs)
- In-batch negatives: All other embeddings in the batch serve as negatives
- Standard approach from SimCLR, CLIP, and modern contrastive learning literature

**Why This Approach**:
- **Siamese Training**: Standard methodology for similarity learning (face recognition, sentence embeddings, image retrieval)
- **Function-Level Matching**: More precise than project-level similarity; directly addresses BCSD task
- **In-Batch Negatives**: Computationally efficient, no hard negative mining required
- **MLM + Contrastive**: MLM ensures instruction-level semantics, contrastive ensures function-level similarity alignment

**Alternatives Considered**:
- **Triplet Loss**: Requires explicit hard negative mining (computationally expensive, complex implementation)
- **Binary Cross-Entropy**: Treats similarity as binary classification, less nuanced than contrastive learning
- **Only MLM**: Doesn't directly optimize for similarity detection task
- **Only Contrastive**: Loses fine-grained semantic understanding of individual instructions
- **Dual-Branch Architecture**: Unnecessary architectural duplication; Siamese training achieves the same goal with simpler code and fewer parameters

**Best Practices**:
- Use AdamW optimizer (weight_decay=0.01 for regularization)
- Learning rate: 2e-5 with linear warmup (10% of training steps)
- Batch size: 16 (provides 8 positive pairs + 14 in-batch negatives per batch, balances memory and gradient stability)
- Save checkpoints every epoch, keep best based on validation loss
- Implement early stopping (patience=3 epochs)
- Monitor contrastive loss separately from MLM loss to ensure both components are learning

---

## 5. Data Representation and Batching

### Decision: Heterogeneous Batching with Custom Collate Function

**Challenge**: Each binary has different sequence length and graph size.

**Solution**:
1. **Sequence Padding**: Pad instruction sequences to max length in batch
   - Create attention masks (1 for real tokens, 0 for padding)
   - Use dynamic padding (not fixed max_length) to save memory
2. **Graph Batching**: Use PyTorch Geometric's Batch class
   - Keeps graphs separate (doesn't merge into single adjacency matrix)
   - Maintains node-to-graph mapping for correct pooling

**Data Format**:
- **Preprocessed Files**: One JSON file per binary
  ```json
  {
    "binary_name": "clamav_clamscan",
    "cfg": {
      "nodes": [{"id": 0, "addr": "0x401000", "instructions": [...]}],
      "edges": [[0, 1], [0, 2], ...]
    },
    "instruction_sequence": ["mov", "rax", ",", "rdi", "call", ...],
    "metadata": {"source": "clamav", "hash": "abc123..."}
  }
  ```

**Dataset Structure**:
- Training: clamav, curl, nmap, openssl (all binaries from these projects)
- Validation: unrar (all binaries)
- Test: z3, zlib (held out, never seen during training)

**Best Practices**:
- Implement DataLoader with num_workers=4 for parallel loading
- Use pin_memory=True for faster GPU transfer
- Implement dataset caching (preload small datasets to RAM)
- Log dataset statistics (sequence length distribution, graph size distribution)

---

## 6. Evaluation Metrics

### Decision: Multi-faceted Evaluation

**Primary Metric: Retrieval Accuracy**
- For each test binary, retrieve top-10 most similar binaries from training set
- Accuracy = % of test binaries where top-1 result is from same project
- Mean Average Precision (MAP) across all test queries

**Secondary Metrics**:
- Validation loss (MLM + contrastive) to monitor overfitting
- Embedding space visualization (t-SNE) to check cluster quality
- Attention weight analysis (how much tokens attend to graph prefix)

**Baseline Comparisons**:
- BERT-only: Use same BERT without GNN component
- GNN-only: Use only graph embeddings without sequence information
- Random: Sanity check (should be ~14% accuracy for 7 projects)

**Best Practices**:
- Report results with confidence intervals (3 runs with different seeds)
- Include confusion matrix showing which projects are confused
- Analyze failure cases (binaries that are misclassified)

---

## 7. Development and Documentation Tools

### Decision: Jupyter Notebook for Exploration + Shell Scripts for Session Management

**Exploration Notebook** (`exploration.ipynb`):
- 4 phases corresponding to pipeline stages
- Each phase independently executable with dummy data
- Visualizations for debugging (CFG graphs, attention matrices, embeddings)
- Serves as both development tool and thesis documentation

**Session Management**:
- Shell script `start_session.sh` creates dated Markdown file in `protocols/`
- Template includes sections: Goals, Actions, Outcomes, Issues
- `end_session.sh` prompts for summary and extracts key points to LaTeX
- LaTeX document organized chronologically for methodology chapter

**Review Agents**:
- Code critic: Rule-based checks (PEP 8, docstrings, error handling)
- Document critic: Check for clarity, jargon, flow (simple heuristics + manual review)
- Not using LLM-based agents to keep complexity low

**Best Practices**:
- Commit session notes to Git for full history
- Use consistent LaTeX formatting (subsections per session)
- Include code snippets and figures in thesis document
- Maintain separate `experiments/` folder for failed approaches (learning value)

---

## 8. Reproducibility Infrastructure

### Decision: Comprehensive Environment and Seed Management

**Environment**:
- `requirements.txt` with pinned versions (pip freeze output)
- Docker container optional but recommended (Ubuntu 22.04 + CUDA 11.8)
- Document GPU model and driver version in README

**Seed Control**:
```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Logging**:
- All hyperparameters logged to config.json
- Training logs (loss per step) saved to CSV
- Model checkpoints include optimizer state for exact resumption
- Git commit hash recorded in experiment logs

**Best Practices**:
- Document any manual steps (dataset download, preprocessing)
- Include expected runtime on reference hardware
- Provide sample outputs for each pipeline stage
- Create comprehensive README with quickstart guide

---

## Research Questions Resolved

1. **Which GNN architecture for CFG?** → GAT for attention-based neighbor weighting
2. **How to fuse graph and sequence?** → KV-Prefix attention mechanism
3. **MLM or contrastive loss?** → Both, in two-stage training
4. **How to handle varying graph sizes?** → PyTorch Geometric batching + attention pooling
5. **Which BERT variant?** → bert-base-uncased for balance of capacity and efficiency
6. **How to ensure reproducibility?** → Seed control + pinned dependencies + comprehensive logging

---

## Next Steps (Phase 1)

1. Define data model (entities, relationships)
2. Specify API contracts between modules
3. Create quickstart guide for setup
4. Update agent context with technology stack
5. Re-evaluate constitution compliance after design phase
