# Implementation Plan: Complete BCSD Pipeline Implementation

**Branch**: `001-bcsd-pipeline-implementation` | **Date**: 2025-12-13 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-bcsd-pipeline-implementation/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement a complete Binary Code Similarity Detection (BCSD) system that integrates Control Flow Graph structure via Graph Neural Networks with sequence-based BERT embeddings. The hypothesis is that this joint model (GNN + KV-Prefix BERT) will achieve higher similarity detection accuracy on test datasets compared to sequence-only or graph-only baselines. The system includes modular preprocessing (angr-based CFG extraction), PyTorch dataset handling, custom attention mechanism for graph-text fusion, training infrastructure with joint MLM+contrastive loss, exploration notebook for iterative development, session management for research documentation, and LaTeX document generation for thesis writing.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: 
- angr (binary analysis, CFG extraction)
- PyTorch 2.0+ (deep learning framework)
- PyTorch Geometric (GNN implementation)
- Hugging Face Transformers (pretrained BERT models)
- NetworkX (graph data structures)
- Matplotlib/Seaborn (visualization)
- Jupyter Notebook (exploration/development)

**Storage**: File-based (JSON for CFG data, Pickle for Python objects, CSV for training logs, PyTorch .pt for model checkpoints)  
**Testing**: pytest (unit tests), manual validation via Jupyter notebook for integration testing  
**Target Platform**: Linux (Ubuntu/Debian) with NVIDIA GPU support (CUDA 11.8+)  
**Project Type**: Research pipeline (single project with modular components)  
**Performance Goals**: 
- CFG extraction: <5 minutes per binary
- GNN processing: <1 second per graph (up to 1000 nodes) on GPU
- Training: 10 epochs in <24 hours on single GPU
- Inference: <5 minutes per binary (including preprocessing)

**Constraints**: 
- Memory: Must handle batches of 16 samples on systems with 16GB RAM
- GPU: Minimum 6GB VRAM for training
- Dataset: 7 binary packages (4 train, 1 validation, 2 test) from Dataset-1
- Reproducibility: Results must be reproducible within 5% variance using fixed random seeds

**Scale/Scope**: 
- Research project for Bachelor's thesis
- ~10 Python modules (preprocessing, dataset, GNN, attention, BERT integration, training, inference, session management, review agents)
- ~1000-2000 lines of core ML pipeline code
- 10 user stories covering technical pipeline + research infrastructure
- Target: 20+ page thesis document with methodology, experiments, results

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Verify compliance with constitution principles (`.specify/memory/constitution.md`):

- [x] **Research Documentation First**: Hypothesis is clearly documented in spec.md (GNN+BERT integration improves similarity detection). Each user story includes independent test criteria. Session management system (US9) ensures all experimental work is documented as it happens with protocols/ directory and LaTeX integration.
- [x] **Reproducible Pipeline**: Each component (preprocessing, dataset, GNN, attention, BERT, training, inference) has clear acceptance scenarios for independent execution. FR-005 ensures preprocessing is offline/standalone. FR-045 ensures notebook sections are independently executable. Dependencies pinned (A-004, D-001 through D-008).
- [x] **Experiment Tracking**: Training module (FR-026 to FR-033) logs all metrics (loss values, validation results) to CSV. Session management (FR-046 to FR-051) tracks research decisions. SC-016 requires reproducibility within 5% variance using fixed seeds.
- [x] **Modular Architecture**: 10 user stories represent distinct modules with single responsibilities (US1: preprocessing, US2: dataset, US3: GNN, US4: attention, US5: BERT integration, etc.). FR requirements grouped by module. Each module has clear interfaces (JSON for CFG data, PyTorch tensors for model components).
- [x] **Scientific Rigor**: Hypothesis stated upfront with expected outcome. Success criteria (SC-001 to SC-016) are measurable. Baseline comparison required (SC-006: BERT-only baseline). Test datasets held out (z3, zlib). Exploration notebook (US6) enables systematic validation at each stage.
- [x] **Context7 Integration**: Will use Context7 MCP tools for retrieving PyTorch, PyTorch Geometric, Transformers, angr documentation during implementation. Code generation for GNN architectures, attention mechanisms, and BERT integration will leverage Context7.

**GATE STATUS**: ✅ **PASSED** - All constitution principles are addressed in the specification. No violations requiring justification.

## Project Structure

### Documentation (this feature)

```text
specs/001-bcsd-pipeline-implementation/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output: Technical decisions & rationale
├── data-model.md        # Phase 1 output: Entity definitions & relationships
├── quickstart.md        # Phase 1 output: Setup & usage guide
├── contracts/           # Phase 1 output: Module API specifications
│   ├── preprocessing_module.md
│   ├── dataset_module.md
│   ├── model_module.md
│   └── training_inference_module.md
├── checklists/
│   └── requirements.md  # Quality validation (completed)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT YET CREATED)
```

### Source Code (repository root)

**Selected Structure**: Single research project with modular pipeline components

```text
BCSD-Model-using-GNN-to-enrichen-V-K-/
├── preprocessing/          # PHASE 1: Data Generation (angr + tokenization)
│   ├── __init__.py
│   ├── extract_features.py         # US1: angr CFG extraction + disassembly
│   ├── tokenizer.py                # US1: Assembly instruction tokenizer (vocab, masking)
│   └── batch_preprocess.py         # Batch processing script for Dataset-1
│
├── dataset/                # PHASE 2: Data Loading
│   ├── __init__.py
│   ├── code_dataset.py             # US2: PyTorch Dataset for loading preprocessed data
│   ├── collate.py                  # US2: Custom collate_fn for heterogeneous batching
│   └── pairing.py                  # US2: Generate positive pairs for Siamese training
│
├── models/                 # PHASE 3: Neural Network Architecture
│   ├── __init__.py
│   ├── gnn_encoder.py              # US3: GAT encoder for graph summarization
│   ├── custom_attention.py         # US4: KV-Prefix attention mechanism (HARD PART)
│   ├── bert_encoder.py             # US4: BERT with custom attention integration
│   └── bcsd_model.py               # US5: Main model connecting GNN→Projection→BERT
│
├── training/               # PHASE 4: Training Infrastructure
│   ├── __init__.py
│   ├── trainer.py                  # US7: Training loop with checkpointing
│   ├── losses.py                   # US7: MLM + InfoNCE contrastive loss
│   └── metrics.py                  # US7: Validation metrics
│
├── inference/              # PHASE 5: Vectorization & Similarity Search
│   ├── __init__.py
│   ├── vectorizer.py               # US8: Convert binaries to embeddings
│   ├── similarity.py               # US8: Cosine similarity search (numpy)
│   └── clustering.py               # US8: Optional k-means/DBSCAN clustering
│
├── utils/                  # Shared Utilities
│   ├── __init__.py
│   ├── reproducibility.py          # Seed setting, deterministic behavior
│   ├── logging.py                  # Structured logging setup
│   └── validation.py               # Input validation helpers
│
├── scripts/                # Executable Entry Points
│   ├── preprocess_binaries.py      # Run preprocessing on Dataset-1
│   ├── train_model.py              # Run training with config
│   ├── run_inference.py            # Run vectorization + similarity search
│   ├── start_session.sh            # US9: Session management
│   ├── end_session.sh              # US9: Session management
│   └── review_document.py          # US10: Document critic (LaTeX)
│
├── configs/                # Configuration files
│   ├── train_config.yaml           # Training hyperparameters
│   └── model_config.yaml           # Model architecture config
│
├── data/                   # Data storage
│   ├── metadata.csv                # Master index: file_hash, project, function_name, optimization, compiler, split_set
│   └── preprocessed/               # Single JSON per function: {hash}.json with tokens + edges
│
├── embeddings/             # Computed embeddings
│   ├── graph_summaries/            # GNN output (.npy files)
│   ├── binary_embeddings/          # Final BERT embeddings (.npy files)
│   └── embedding_database.pkl      # Serialized embedding DB for inference
│
├── checkpoints/            # Model checkpoints
│   └── model_epoch_{N}_valloss_{loss}.pt
│
├── logs/                   # Training logs
│   ├── training_metrics.csv        # Per-step metrics
│   └── experiment_{date}.log       # Detailed logs
│
├── protocols/              # Research session logs (US9)
│   ├── session_2025-12-13.md
│   └── ...
│
├── thesis/                 # LaTeX thesis document (US9)
│   ├── main.tex
│   ├── methodology.tex             # Extracted from session logs
│   ├── results.tex
│   └── figures/
│
├── test_binaries/          # Controlled test binaries for validation
│   ├── test_gnn.c                  # Source: Simple functions with loops/branches
│   ├── compile.sh                  # Compilation script (gcc/clang variants)
│   ├── README.md                   # Test binary documentation
│   ├── test_gnn_gcc_O0             # Compiled: gcc -O0 (no optimization)
│   ├── test_gnn_gcc_O3             # Compiled: gcc -O3 (full optimization)
│   ├── test_gnn_clang_O0           # Compiled: clang -O0
│   ├── test_gnn_clang_O3           # Compiled: clang -O3
│   └── expected_outputs/           # Expected CFG/tokenization results
│       ├── test_gnn_gcc_O0_cfg.json
│       └── test_gnn_gcc_O0_tokens.txt
│
├── tests/                  # Test suite
│   ├── test_preprocessing.py       # Includes test_binaries validation
│   ├── test_dataset.py
│   ├── test_gnn.py
│   ├── test_bert_integration.py
│   ├── test_training.py
│   └── test_inference.py
│
├── exploration.ipynb       # US6: Thesis demonstration notebook (publication-quality figures)
├── requirements.txt        # Pinned dependencies
├── README.md               # Project overview
├── SETUP_GUIDE.md          # Detailed setup (references quickstart.md)
└── .gitignore

Dataset-1/                  # External (not in repo)
├── clamav/
├── curl/
├── nmap/
├── openssl/
├── unrar/
├── z3/
└── zlib/
```

**Structure Decision**: 

This is a **phase-based research project** structure organized by data flow:

1. **Clarity**: Each phase represents a distinct stage in the pipeline (data generation → loading → modeling → training → inference)
2. **Modularity**: Clear separation between feature extraction (preprocessing), data handling (dataset), architecture (models), training, and inference
3. **Explicit Tokenization**: `preprocessing/tokenizer.py` handles assembly-specific tokenization (vocab building, masking) separate from BERT's tokenizer
4. **Custom Attention Isolation**: `models/custom_attention.py` is the critical research contribution - isolated for focused development
5. **God Class Pattern**: `models/bcsd_model.py` orchestrates GNN→Projection→BERT integration
6. **Independent Testing**: Each phase can be tested in isolation
7. **Reproducibility**: Configurations, checkpoints, and logs tracked separately

**Phase Mapping to User Stories**:
- **Phase 1 (preprocessing/)**: US1 - angr CFG extraction + assembly tokenization
- **Phase 2 (dataset/)**: US2 - PyTorch Dataset + Siamese pair generation + heterogeneous batching
- **Phase 3 (models/)**: US3, US4, US5 - GNN encoder, custom KV-Prefix attention, BERT integration, main BCSD model
- **Phase 4 (training/)**: US7 - Training loop, MLM + contrastive loss, checkpointing
- **Phase 5 (inference/)**: US8 - Vectorization, similarity search, clustering
- **Research Tools**: US6 (exploration.ipynb), US9 (session management), US10 (review agents)

**Key Architectural Highlights**:
- `preprocessing/tokenizer.py`: Custom assembly tokenizer (NOT BERT's tokenizer) - builds vocab from opcodes/operands
- `models/custom_attention.py`: KV-Prefix attention - the "HARD PART" where graph summary injects into BERT
- `models/bcsd_model.py`: Main integration point - connects all components
- `training/losses.py`: Joint loss (MLM + InfoNCE) for Siamese training
- `inference/similarity.py`: Pure numpy/scipy similarity search (no neural network)

## Complexity Tracking

> **No violations - This section is empty per constitution compliance**

All constitution principles are satisfied by the design:
- Research documentation is integrated via session management and LaTeX generation
- Pipeline stages are independently executable and testable
- Experiment tracking is built into training infrastructure
- Modular architecture with clear module boundaries
- Scientific rigor enforced through validation metrics and baseline comparisons
- Context7 will be used for implementation guidance

No additional complexity or violations requiring justification.
