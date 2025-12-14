# Feature Specification: Complete BCSD Pipeline Implementation

**Feature Branch**: `001-bcsd-pipeline-implementation`  
**Created**: 2025-12-13  
**Status**: Draft  
**Input**: User description: "Implement complete BCSD pipeline with GNN-enriched BERT including modular preprocessing, dataset handling, custom attention mechanism, training infrastructure, exploration notebook, session management, and documentation system"

<!--
  NOTE: This is a Bachelor's thesis research project. User stories represent implementation
  phases that build toward a complete end-to-end system while maintaining modularity for
  experimentation and validation at each stage.
-->

## User Scenarios & Testing *(mandatory)*

<!--
  Research Project Context: Each user story represents an independently testable milestone
  in building the complete BCSD pipeline. Stories are ordered by dependency and research value.
-->

### User Story 1 - Binary Preprocessing with angr + Custom Tokenization (Priority: P1)

As a researcher, I need a preprocessing module that:
1. Extracts Control Flow Graphs (CFG) from binaries using angr
2. Extracts disassembled instruction sequences
3. Applies custom assembly tokenization (building vocab from opcodes/operands, not BERT's tokenizer)

...so that I can generate structured, tokenized input data for the machine learning pipeline without blocking the training loop.

**Why this priority**: This is the foundational data generation layer (PHASE 1). Without preprocessed + tokenized data, no other components can be developed or tested. Custom tokenization is critical because assembly language has domain-specific structure (opcodes, registers, immediates) that BERT's word-piece tokenizer cannot handle properly.

**Independent Test**: Can be fully tested by:
1. Running `preprocessing/extract_features.py` on a binary → outputs CFG JSON + disassembly JSON
2. Running `preprocessing/tokenizer.py` on disassembly → outputs tokenized sequences with vocab mappings
3. Verifying CFG structure (nodes, edges) and token sequences (opcode/operand separation)

**Acceptance Scenarios**:

1. **Given** the `test_binaries/test_gnn` compiled binary (containing simple functions with loops and branches), **When** the preprocessing script is executed, **Then** it produces a JSON file containing CFG edges showing loop cycles, branch points, and linearized instruction sequences for all three functions (calculate_sum, check_value, main)
2. **Given** the test binary CFG output JSON, **When** loaded and inspected, **Then** it contains:
   - Valid graph edge lists showing the loop in calculate_sum (backward edge creating cycle)
   - Branch nodes in check_value (if/else creating multiple paths)
   - Tokenized assembly instructions with opcodes and operands for each basic block
   - Function metadata (names: calculate_sum, check_value, main; addresses; block counts)
3. **Given** an unstripped ELF binary from the training dataset (clamav, curl, nmap, openssl), **When** the preprocessing script is executed, **Then** it produces a JSON file containing CFG edges, basic block information, and linearized instruction sequences
4. **Given** a binary with missing symbols or corrupted sections, **When** preprocessing is attempted, **Then** the system logs warnings and continues processing available sections without crashing
5. **Given** multiple binaries processed in batch mode, **When** preprocessing completes, **Then** each output file is named consistently (e.g., `binary_name_cfg.json`) and stored in the designated output directory

---

### User Story 2 - PyTorch Dataset Implementation (Priority: P2)

As a researcher, I need a PyTorch Dataset class that loads preprocessed CFG data and handles batching of heterogeneous data types (padded token sequences + graph adjacency matrices), so that I can efficiently feed data into the training loop without memory issues or shape mismatches.

**Why this priority**: Once preprocessing is working, we need proper data loading infrastructure before implementing models. This enables parallel development of GNN and BERT components.

**Independent Test**: Can be fully tested by instantiating the Dataset with preprocessed files, iterating through batches using a DataLoader, and verifying that each batch contains properly padded sequences and packed graph structures with consistent dimensions. Success means no shape errors during iteration.

**Acceptance Scenarios**:

1. **Given** preprocessed JSON files in a directory, **When** the Dataset is instantiated, **Then** it successfully indexes all files and reports the total number of samples
2. **Given** a batch size of 16, **When** iterating through the DataLoader, **Then** each batch contains padded instruction sequences (shape: [batch_size, max_seq_len]) and a list of graph objects (one per sample) with edge indices
3. **Given** samples with varying sequence lengths, **When** collate_fn is applied, **Then** shorter sequences are padded to match the longest in the batch, and a padding mask is provided
4. **Given** graphs with different numbers of nodes, **When** batched together, **Then** they are handled as separate graph objects (not forced into a single adjacency matrix) to preserve structure

---

### User Story 3 - GNN Encoder for Graph Summarization (Priority: P3)

As a researcher, I need a Graph Neural Network encoder that processes CFG structures and produces fixed-size graph summaries (Graph Summary vector), so that structural information from binary code can be integrated with BERT's text-based embeddings.

**Why this priority**: With data loading working, we can now implement the GNN component independently. This is a critical research component as it captures the unique graph structure of binary code.

**Independent Test**: Can be fully tested by feeding a batch of graph objects (edge indices, node features) into the GNN encoder and verifying that the output is a fixed-size tensor (e.g., [batch_size, hidden_dim]) representing the graph summary via global pooling. Test with graphs of varying sizes.

**Acceptance Scenarios**:

1. **Given** a single CFG graph with N nodes and E edges, **When** passed through the GNN encoder, **Then** it outputs a fixed-size vector (e.g., 256-dimensional) regardless of N and E
2. **Given** a batch of graphs with varying node counts (e.g., 10, 50, 200 nodes), **When** processed by the GNN, **Then** all outputs have the same dimensionality and the batch dimension is preserved
3. **Given** node features initialized from instruction embeddings, **When** message passing occurs over multiple GNN layers, **Then** the receptive field expands to capture multi-hop neighborhood information
4. **Given** an isolated node (no edges), **When** processed, **Then** the GNN handles it gracefully and produces a valid summary based on the node's own features

---

### User Story 4 - Custom KV-Prefix Attention Mechanism (Priority: P4)

As a researcher, I need a custom attention layer that injects graph-derived Keys and Values as a prefix into BERT's attention mechanism, so that the model can attend to both sequential instruction tokens and structural graph information simultaneously during training.

**Why this priority**: This is the novel research contribution - the fusion mechanism. It requires both GNN and BERT understanding, so it comes after both are individually testable.

**Independent Test**: Can be fully tested in isolation by creating dummy Query, Key, Value tensors from BERT, dummy Graph Summary from GNN, passing them through the custom attention layer, and verifying that the output attention matrix has the correct dimensions (sequence_length + prefix_length) and that gradients flow correctly.

**Acceptance Scenarios**:

1. **Given** a Graph Summary vector of dimension [batch_size, graph_dim], **When** projected into Keys and Values, **Then** the resulting tensors have shape [batch_size, num_heads, 1, head_dim] suitable for concatenation with sequence Keys/Values
2. **Given** text Keys/Values of shape [batch_size, num_heads, seq_len, head_dim] and graph-derived prefix Keys/Values, **When** concatenated, **Then** the combined Keys/Values have shape [batch_size, num_heads, seq_len+1, head_dim]
3. **Given** Query tokens attending to combined Keys, **When** attention is computed, **Then** each token can attend to both the graph prefix and all other tokens in the sequence
4. **Given** a forward and backward pass through the attention layer, **When** gradients are computed, **Then** gradients flow back to both the graph summary projection and the text embeddings

---

### User Story 5 - Siamese Training with Joint MLM and Contrastive Loss (Priority: P5)

As a researcher, I need a unified model class that combines the GNN encoder, custom attention mechanism, and BERT backbone to enable Siamese-style training where a single model (shared weights) processes multiple function pairs to generate embeddings, optimized with joint MLM and contrastive loss for binary code similarity detection.

**Why this priority**: This integrates all previous components into a trainable system using Siamese architecture. It's the culmination of the pipeline and requires all prior stories to be complete.

**Architecture Note**: The model is a single "tower" (GNN + BERT) with one input interface, NOT two separate input branches. During training, the same model instance with shared weights processes multiple inputs (function pairs) sequentially within a batch to generate multiple embeddings that are then compared using contrastive loss.

**Independent Test**: Can be fully tested by: (1) Creating a batch with positive pairs (same function compiled differently) and in-batch negatives, (2) Processing each sample through the model to generate N embeddings, (3) Computing joint loss (MLM for each sample + contrastive comparing all embeddings), (4) Running backward pass and verifying gradients update both GNN and BERT components.

**Acceptance Scenarios**:

1. **Given** a batch of N functions (containing positive pairs like func_gcc and func_clang), **When** each is processed through the same model instance, **Then** it outputs N embeddings (shape [N, 768]) and N sets of MLM predictions without errors
2. **Given** embeddings from positive pairs (semantically identical functions) and in-batch negatives (different functions), **When** contrastive loss is computed, **Then** it correctly minimizes distance for positive pairs and maximizes distance for negatives using InfoNCE/NT-Xent approach
3. **Given** masked tokens in each input sequence, **When** MLM loss is computed independently for each sample, **Then** the model predicts masked assembly instructions with improving accuracy
4. **Given** a training loop of 10 steps with Siamese batching, **When** losses are logged each step, **Then** both MLM and contrastive losses decrease, confirming the model learns both instruction semantics and similarity relationships
5. **Given** a trained model checkpoint, **When** saved and reloaded, **Then** inference produces identical embeddings for the same inputs, demonstrating deterministic feature extraction

---

### User Story 6 - Thesis Demonstration Notebook (Priority: P6)

As a researcher, I need a Jupyter notebook with publication-quality demonstrations of each pipeline component (preprocessing, GNN, attention, full model) with visualizations for thesis figures, so that I can generate clear illustrations of the system for my thesis documentation.

**Why this priority**: This is focused on thesis presentation, not development tooling. Creates figures and demonstrations for academic documentation.

**Independent Test**: Run notebook sections to generate thesis figures: CFG visualization, GNN graph summary heatmap, attention matrix visualization, and embedding t-SNE plot. Success means generating publication-ready figures suitable for thesis inclusion.

**Acceptance Scenarios**:

1. **Given** the notebook sections for each component, **When** executed with test data, **Then** it generates clear, labeled visualizations (CFG graph, attention heatmap, embedding plots) suitable for thesis figures
2. **Given** the complete notebook, **When** run end-to-end, **Then** it produces a comprehensive demonstration showing data flow from binary → CFG → embeddings → similarity scores

---

### User Story 7 - Training Script and Infrastructure (Priority: P7)

As a researcher, I need a training script that loads preprocessed data, trains the model with the joint loss function, logs metrics (loss curves, accuracy), and saves checkpoints at intervals, so that I can conduct experiments systematically and track model performance over time.

**Why this priority**: With all components integrated, this enables actual research experiments. It's prioritized after the exploration notebook because debugging should happen interactively first.

**Independent Test**: Can be fully tested by running a training session on a small subset of data (e.g., 100 samples from training datasets) for 10 epochs, verifying that loss decreases, checkpoints are saved, and training logs are written to disk. Success means completing training without crashes and producing reproducible results.

**Acceptance Scenarios**:

1. **Given** preprocessed data from the first 4 datasets (clamav, curl, nmap, openssl), **When** training begins, **Then** the script loads all samples into memory-mapped or streamed format and reports dataset statistics
2. **Given** a training configuration (batch size=16, epochs=10, learning rate=1e-4), **When** training runs, **Then** loss (both MLM and contrastive) is logged every N steps and saved to a CSV file
3. **Given** training in progress, **When** each epoch completes, **Then** a model checkpoint is saved with epoch number in the filename (e.g., `model_epoch_05.pt`)
4. **Given** validation data (unrar dataset), **When** validation runs after each epoch, **Then** validation loss is computed and logged separately from training loss, enabling early stopping decisions
5. **Given** a crash during training, **When** restarted from the last checkpoint, **Then** training resumes from the saved state without recomputing earlier epochs

---

### User Story 8 - Vectorization and Similarity Search (Priority: P8)

As a researcher, I need an inference pipeline that uses the trained model as a feature extractor to convert binaries into fixed-size embedding vectors (fingerprints), then performs mathematical similarity search using cosine similarity or Euclidean distance to identify similar functions and cluster binary families, so that I can detect similar code without neural network computation at query time.

**Why this priority**: This demonstrates the complete system's practical application: converting code into vectors and using classical similarity metrics for detection. It's essential for thesis demonstration and shows how the learned representations enable efficient similarity search.

**Architecture Note**: The trained model serves as a **Feature Extractor**, not a classifier. During inference, it translates "Code + Graph" into a fixed-size vector (e.g., [768]). The actual similarity detection happens **outside the neural network** using simple mathematical distance metrics.

**Independent Test**: Can be fully tested by: (1) Processing multiple test binaries to generate embedding vectors, (2) Computing pairwise cosine similarities between vectors using numpy/scipy, (3) Identifying top-K similar functions based on distance, (4) Optionally clustering vectors to discover binary families. Success means completing vectorization and similarity search in reasonable time (<5 minutes total).

**Acceptance Scenarios**:

1. **Given** a folder of raw binary files from the test set, **When** the vectorization script runs, **Then** it automatically preprocesses each binary using angr and generates embedding vectors (shape [num_functions, 768]) saved to disk
2. **Given** a query function's embedding vector and a database of reference embeddings, **When** similarity search is performed using cosine similarity, **Then** it returns a ranked list of the top-10 most similar functions with similarity scores in range [-1, 1]
3. **Given** embeddings from multiple binaries in a test dataset, **When** clustering is performed using distance metrics (e.g., k-means or DBSCAN on cosine distances), **Then** the system groups similar functions into clusters representing binary families or variants
4. **Given** a corrupted binary that fails preprocessing, **When** vectorization is attempted, **Then** the script logs a warning, skips the file, and continues processing remaining binaries without crashing
5. **Given** a precomputed embedding database and a new query binary, **When** similarity search runs, **Then** it completes in <1 second using pure numpy/scipy operations (no GPU needed), demonstrating efficient retrieval

---

### User Story 9 - Session Management System (Priority: P9)

As a researcher, I need a session management system that creates Markdown files for each work session, logs progress and decisions, and compresses session notes into a continuously updated LaTeX document for thesis writing, so that my research process is fully documented from day one.

**Why this priority**: This is a documentation infrastructure requirement that supports the research process but doesn't affect the core pipeline. It can be implemented in parallel with technical work.

**Independent Test**: Can be fully tested by starting a session (creating a new MD file), logging several research activities, ending the session, and verifying that important points are extracted and appended to the main LaTeX file. Success means readable, structured documentation.

**Acceptance Scenarios**:

1. **Given** the command "start today's session", **When** executed, **Then** a new Markdown file is created in `protocols/` directory with the current date and a session template (sections for goals, actions, outcomes, issues)
2. **Given** an active session MD file, **When** work is performed (code written, experiments run), **Then** the researcher manually logs key activities, and the file accumulates timestamped entries
3. **Given** the command "end today's session", **When** executed, **Then** the system prompts for a summary, extracts key learnings/decisions, and appends them to the main LaTeX file in a structured format (subsection per session)
4. **Given** the main LaTeX document, **When** reviewed, **Then** it contains chronological summaries of all sessions, written in academic style suitable for a methodology chapter, with clear explanations of tools and decisions

---

### User Story 10 - Document Review Agent (Priority: P10)

As a researcher, I need a document review agent that analyzes LaTeX sections and provides feedback on writing quality (clarity, technical explanations, academic tone), so that I maintain high-quality thesis documentation.

**Why this priority**: Focuses on thesis writing quality. Code quality will be handled through standard tools (pylint, black) rather than custom agents.

**Independent Test**: Submit LaTeX section to document critic, receive specific feedback on clarity, flow, and academic style. Success means receiving actionable suggestions with specific references.

**Acceptance Scenarios**:

1. **Given** a LaTeX section from the thesis document, **When** submitted to the document critic agent, **Then** it returns feedback on: paragraph flow, technical term explanations, sentence complexity, repetition, and appropriate formality for academic writing
2. **Given** feedback suggesting improvements, **When** reviewed, **Then** each point references specific sentences/paragraphs and explains why the suggestion improves readability for a student audience (target: understandable to peers)

**Note**: Code quality will be maintained using standard tools (pylint, black, flake8) rather than a custom code critic agent.

---

### Edge Cases

- What happens when a binary file is corrupted or unreadable during preprocessing? System logs error and skips the file.
- What happens when the CFG has disconnected components (multiple entry points)? GNN processes each component separately via batching.
- What happens when a batch contains only one sample? Batch normalization and contrastive loss need special handling (e.g., skip contrastive for batch=1).
- What happens when disk space runs out during checkpoint saving? Training fails gracefully with clear error message.
- What happens when a session is started but not properly closed? The MD file remains incomplete; LaTeX file is not updated until manual cleanup.
- What happens when two similar binaries have different compilation options (optimization levels)? This is expected behavior; model should learn to focus on semantic similarity rather than exact instruction matching.

## Requirements *(mandatory)*

### Hypothesis

**Central Hypothesis**: Integrating Control Flow Graph structure via Graph Neural Networks with sequence-based BERT embeddings will improve binary code similarity detection accuracy compared to sequence-only or graph-only approaches.

**Expected Outcome**: The joint model (GNN + KV-Prefix BERT) will achieve higher similarity detection accuracy on the test datasets (z3, zlib) compared to baseline models (BERT-only, GNN-only).

### Functional Requirements

#### Preprocessing Module

- **FR-001**: System MUST use angr's CFGFast with `auto_load_libs=False` to extract Control Flow Graphs from ELF binaries
- **FR-002**: System MUST output CFG data as JSON files containing node IDs, edge lists (source, target), and basic block metadata (addresses, instruction counts)
- **FR-003**: System MUST linearize assembly instructions into token sequences suitable for BERT tokenization
- **FR-004**: System MUST handle binaries with missing or stripped symbols gracefully by using address-based identification
- **FR-005**: Preprocessing MUST be an offline, standalone process that outputs serialized data (JSON/Pickle), not integrated into the training loop

#### Dataset Module

- **FR-006**: System MUST implement a PyTorch Dataset class that loads preprocessed CFG data from disk
- **FR-007**: DataLoader MUST use a custom collate_fn that pads instruction sequences to the maximum length in each batch
- **FR-008**: Collate function MUST preserve graph structures as separate objects (not merged into a single adjacency matrix)
- **FR-009**: Dataset MUST provide padding masks indicating valid tokens vs. padding tokens for attention masking
- **FR-010**: System MUST support splitting data into training (clamav, curl, nmap, openssl), validation (unrar), and test (z3, zlib) sets

#### GNN Module

- **FR-011**: System MUST implement a Graph Neural Network encoder that processes CFG edge lists and node features
- **FR-012**: GNN MUST use message passing (e.g., Graph Convolutional Networks or Graph Attention Networks) over multiple layers
- **FR-013**: GNN MUST output fixed-size graph summaries via global pooling (e.g., mean pooling or attention-based pooling)
- **FR-014**: Graph summary dimension MUST be configurable for experimentation (default: 256-dimensional vector, can be adjusted to 128, 256, or 512 based on experiments)
- **FR-015**: GNN MUST handle graphs of varying sizes (10 to 1000+ nodes) efficiently

#### Custom Attention Module

- **FR-016**: System MUST implement a projection layer that maps graph summaries to Key and Value tensors compatible with BERT's attention mechanism
- **FR-017**: Projection MUST produce Keys and Values with shape [batch_size, num_heads, 1, head_dim] for prefix concatenation
- **FR-018**: Attention mechanism MUST concatenate graph-derived Keys/Values as a prefix to sequence Keys/Values before computing attention
- **FR-019**: System MUST ensure Query tokens can attend to both the graph prefix and all sequence tokens in the attention matrix
- **FR-020**: Custom attention MUST maintain compatibility with BERT's multi-head attention architecture (768-dim embeddings, 12 heads)

#### BERT Integration Module

- **FR-021**: System MUST use a pretrained BERT model as the backbone (e.g., `bert-base-uncased`)
- **FR-022**: System MUST replace or modify BERT's attention layers to integrate the custom KV-Prefix attention mechanism
- **FR-023**: BERT MUST process linearized instruction sequences as input tokens
- **FR-024**: System MUST support Masked Language Modeling (MLM) loss for pretraining on binary code semantics
- **FR-025**: Model MUST implement a single "tower" architecture (not dual-branch) where the same model instance with shared weights processes multiple inputs during Siamese training
- **FR-026**: BERT embeddings (CLS token or pooled output) MUST be extractable as fixed-size vectors (768-dim) for similarity computation

#### Training Module

- **FR-027**: System MUST implement a training script that uses Siamese network strategy: the same model instance processes multiple function pairs within each batch
- **FR-028**: Dataset MUST yield positive pairs (semantically identical functions, e.g., func_gcc and func_clang) for contrastive learning
- **FR-029**: Training MUST use a joint loss function: L_total = L_MLM + λ * L_contrast (where λ ∈ {0.3, 0.5, 0.7} is tunable based on validation loss)
- **FR-030**: System MUST implement InfoNCE/NT-Xent contrastive loss that treats matched pairs as "Positives" and all other samples in the batch as "In-Batch Negatives"
- **FR-031**: Contrastive loss MUST minimize cosine distance for positive pairs and maximize distance for negative pairs in embedding space
- **FR-032**: MLM loss MUST be computed independently for each function in the batch to ensure the model understands assembly semantics
- **FR-033**: Training script MUST log loss values (MLM, contrastive, total) after each epoch
- **FR-034**: System MUST save model checkpoints after each epoch with epoch number in the filename
- **FR-035**: Training MUST support resuming from checkpoints in case of interruption
- **FR-036**: System MUST run validation on the validation set (unrar) after each training epoch
- **FR-037**: Training logs MUST be saved to disk in CSV format for analysis

#### Inference Module

- **FR-038**: System MUST provide a vectorization script that uses the trained model as a feature extractor to convert binaries into fixed-size embedding vectors
- **FR-039**: Vectorization script MUST accept a folder of raw binaries and process each function individually to generate embeddings (shape [num_functions, 768])
- **FR-040**: System MUST automatically call preprocessing (angr) for each binary before embedding extraction
- **FR-041**: Embeddings MUST be saved to disk in a structured format (e.g., numpy .npy files or HDF5) for efficient retrieval
- **FR-042**: System MUST provide a similarity search function that computes cosine similarity (or Euclidean distance) between query and reference embeddings using numpy/scipy
- **FR-043**: Similarity search MUST operate without GPU/neural network computation (pure mathematical operations on precomputed vectors)
- **FR-044**: System MUST support top-K retrieval, returning the K most similar functions to a query with similarity scores
- **FR-045**: Results MUST display at least the top-10 most similar functions with source binary names and similarity scores

**Note**: Clustering functionality (k-means, DBSCAN) is optional and moved to Future Work section.

#### Thesis Demonstration Notebook

- **FR-040**: System MUST provide a Jupyter notebook (`demonstration.ipynb`) with sections for generating thesis figures
- **FR-041**: Notebook MUST generate publication-quality CFG visualizations with NetworkX
- **FR-042**: Notebook MUST generate attention matrix heatmaps showing graph prefix attendance
- **FR-043**: Notebook MUST generate embedding space visualizations (t-SNE or PCA plots)
- **FR-044**: Notebook MUST include markdown explanations suitable for thesis methodology chapter

#### Session Management

- **FR-046**: System MUST provide a mechanism to start a new session that creates a Markdown file in `protocols/` directory with a timestamp
- **FR-047**: Session MD files MUST include sections for: session goals, actions taken, outcomes/results, issues encountered
- **FR-048**: System MUST provide a mechanism to end a session that prompts for a summary and key learnings
- **FR-049**: Upon session end, system MUST extract important points from the session MD and append them to the main LaTeX document
- **FR-050**: LaTeX document MUST organize session summaries chronologically as subsections in a methodology or process chapter
- **FR-051**: LaTeX content MUST be written in academic style suitable for a Bachelor's thesis (clear explanations, appropriate formality)

#### Document Review Agent

- **FR-052**: System MUST provide a document review agent that analyzes LaTeX sections and returns feedback on writing quality
- **FR-053**: Document critic MUST check for: paragraph flow, clarity of technical explanations, sentence complexity, repetition, academic tone
- **FR-054**: Agent MUST provide specific, actionable feedback with references to exact text locations

**Note**: Code quality will be ensured through standard tools (pylint for linting, black for formatting, flake8 for style) rather than a custom code critic.

### Key Entities

- **Binary Executable**: An ELF file containing compiled machine code. Attributes: file path, hash, compilation options (gcc/clang, O0/O3), source dataset.
- **Control Flow Graph (CFG)**: A directed graph representing execution flow in a binary. Attributes: nodes (basic blocks), edges (control flow transitions), entry points.
- **Basic Block**: A sequence of instructions with a single entry and single exit. Attributes: start address, end address, instruction count, list of assembly instructions.
- **Instruction Sequence**: Linearized list of assembly instructions extracted from a function for BERT input. Attributes: function name, tokenized opcodes/operands, sequence length.
- **Graph Summary**: A fixed-size vector representation of a CFG produced by the GNN encoder. Attributes: dimension (256), derived from global attention-based pooling.
- **Embedding**: A fixed 768-dimensional vector output from BERT (CLS token or pooled representation) used for similarity computation. Attributes: dimension (768), function identifier, source binary.
- **Training Sample (Positive Pair)**: A pair of functions representing semantically identical code compiled with different settings. Attributes: func_A_embedding, func_B_embedding, label (positive=1), compilation_variant (e.g., gcc_O0 vs clang_O3).
- **In-Batch Negatives**: All other samples in a training batch that are not positive pairs relative to a given anchor function. Used automatically in InfoNCE/NT-Xent loss computation.
- **Contrastive Loss**: InfoNCE/NT-Xent loss computed over positive pairs and in-batch negatives. Formula: -log(exp(sim(anchor, positive)/τ) / Σ exp(sim(anchor, all)/τ)) where τ is temperature.
- **Model Checkpoint**: A saved state of the trained single-tower model (GNN + BERT weights). Attributes: epoch number, model weights, optimizer state, MLM loss, contrastive loss, validation metrics.
- **Similarity Score**: Cosine similarity (or Euclidean distance) computed between two embedding vectors using numpy/scipy operations (no neural network involved).
- **Embedding Database**: Collection of precomputed vectors for all functions in a reference dataset. Attributes: numpy array (shape [N, 768]), function identifiers, stored as .npy files for fast retrieval.
- **Binary Family/Cluster**: Group of similar functions discovered through clustering algorithms (k-means, DBSCAN) applied to embedding vectors. Attributes: cluster_id, member functions, centroid vector.

## Success Criteria *(mandatory)*

### Measurable Outcomes

**Research and Implementation:**

- **SC-001**: Preprocessing module successfully extracts CFG data from 100% of unstripped binaries in Dataset-1 (4 training + 1 validation + 2 test datasets)
- **SC-002**: PyTorch Dataset and DataLoader handle batches of size 16 without memory errors on a system with 16GB RAM
- **SC-003**: GNN encoder processes graphs with up to 1000 nodes in under 1 second per graph on a modern GPU
- **SC-004**: Custom KV-Prefix attention mechanism integrates with BERT without causing dimension mismatches or gradient vanishing
- **SC-005**: Joint training completes 10 epochs on the training datasets (clamav, curl, nmap, openssl) in under 24 hours on a single GPU using Siamese batch construction

**Model Performance:**

- **SC-006**: After training, embeddings from positive pairs (e.g., func_gcc vs func_clang of the same function) have cosine similarity > 0.7
- **SC-007**: On the test datasets (z3, zlib), the model's embeddings cluster semantically similar functions with silhouette score > 0.6
- **SC-008**: Vectorization (embedding extraction) for a single test binary completes in < 5 seconds on GPU or < 30 seconds on CPU
- **SC-009**: Similarity search over 10,000 precomputed vectors completes in < 1 second using numpy cosine similarity (no GPU required)

**Documentation and Research Process:**

- **SC-010**: Exploration notebook contains working examples for all 4 phases (preprocessing, GNN, attention, full model) that execute without errors
- **SC-011**: At least 10 research sessions are documented in the `protocols/` directory with corresponding entries in the main LaTeX file
- **SC-012**: Main LaTeX document reaches at least 20 pages with clear explanations of methodology, tools used, and design decisions
- **SC-013**: Code review agent provides feedback on at least 5 different modules, identifying at least 3 actionable improvements per module
- **SC-014**: Document review agent reviews at least 5 LaTeX sections, identifying clarity issues or jargon that needs explanation

**Reproducibility and Quality:**

- **SC-015**: Another researcher can clone the repository, follow the README, and successfully run preprocessing, training (Siamese batching), and inference (vectorization + similarity search) without needing to ask questions
- **SC-016**: All Python modules include docstrings for public functions, and code passes PEP 8 linting with no errors
- **SC-017**: Training results (loss curves, accuracy metrics) are reproducible within 5% variance when using the same random seed
- **SC-018**: Siamese architecture validation: The same model instance (single tower with shared weights) correctly processes multiple samples per batch, confirmed by parameter identity checks during training

## Assumptions

- **A-001**: The Dataset-1 folder contains unstripped ELF binaries (Linux format) as specified. If binaries are stripped, angr may have reduced CFG accuracy, but this is acceptable for initial experiments.
- **A-002**: A system with at least one NVIDIA GPU (6GB+ VRAM) is available for training. CPU-only training is possible but will take significantly longer.
- **A-003**: Pretrained BERT models (e.g., `bert-base-uncased`) are accessible via Hugging Face Transformers library.
- **A-004**: The researcher has basic familiarity with PyTorch, Git, and Jupyter notebooks. The project will document advanced concepts but not teach fundamentals.
- **A-005**: Session management (start/end commands) will be implemented as simple shell scripts or Python CLI tools, not a complex GUI.
- **A-006**: The review agents will use rule-based analysis or simple heuristics, not advanced LLM-based analysis (unless Context7 or similar tools provide this capability easily).
- **A-007**: The main LaTeX document follows a standard thesis template structure (introduction, methodology, results, discussion, conclusion).

## Dependencies

- **D-001**: angr framework for binary analysis (CFG extraction)
- **D-002**: PyTorch for deep learning infrastructure (Dataset, DataLoader, nn.Module)
- **D-003**: PyTorch Geometric (or similar) for GNN implementations
- **D-004**: Hugging Face Transformers for pretrained BERT models
- **D-005**: NetworkX or similar for graph data structures and visualization
- **D-006**: Matplotlib/Seaborn for plotting (loss curves, attention matrices)
- **D-007**: Jupyter Notebook for exploration.ipynb
- **D-008**: LaTeX distribution (TeX Live or MiKTeX) for thesis document compilation

## Out of Scope

- **OS-001**: Implementing support for non-ELF binary formats (Windows PE, macOS Mach-O) - focus is on Linux ELF binaries only
- **OS-002**: Training on stripped binaries in the initial version - this is a future enhancement
- **OS-003**: Hyperparameter optimization (grid search, Bayesian optimization) - initial experiments will use reasonable defaults
- **OS-004**: Deployment as a web service or CLI tool for end users - the inference script is sufficient for research purposes
- **OS-005**: Extensive ablation studies comparing different GNN architectures (GCN vs GAT vs GraphSAGE) - initial implementation will use one GNN type (to be decided during planning)
- **OS-006**: Real-time binary monitoring or integration with malware analysis platforms - this is pure research, not production software
