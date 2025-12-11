# BCSD Model - Binary Code Similarity Detection

A Binary Code Similarity Detection (BCSD) model that uses Graph Neural Networks (GNN) to enrich BERT embeddings for improved similarity detection between binary executables.

## Overview

This repository implements a complete pipeline for binary code similarity detection using state-of-the-art deep learning techniques:

1. **Angr Disassembly** - Disassembles binary executables and extracts control flow graphs
2. **BERT Encoding** - Encodes assembly instructions and code features into dense vector representations
3. **GNN Processing** - Uses Graph Neural Networks to enrich embeddings with graph structure information

## Project Structure

```
.
├── pipeline/
│   ├── __init__.py           # Package initialization
│   ├── angr_disassembly.py   # Binary disassembly using angr
│   ├── bert_encoder.py       # BERT-based encoding module
│   ├── gnn_model.py          # Graph Neural Network implementation
│   └── pipeline.py           # Main pipeline orchestration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Nguyen-Bang/BCSD-Model-using-GNN-to-enrichen-V-K-.git
cd BCSD-Model-using-GNN-to-enrichen-V-K-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Processing a Single Binary

```bash
python pipeline/pipeline.py /path/to/binary
```

### Processing a Folder of Binaries

```bash
python pipeline/pipeline.py /path/to/binaries/folder
```

### Comparing Two Binaries

```bash
python pipeline/pipeline.py /path/to/binary1 --compare /path/to/binary2
```

### Finding Similar Binaries

```bash
python pipeline/pipeline.py /path/to/binaries/folder --threshold 0.7 --output results.json
```

### Command-Line Options

- `input` - Path to binary file or folder containing binaries
- `--output, -o` - Output file for results (JSON format)
- `--compare` - Path to second binary for comparison
- `--threshold, -t` - Similarity threshold for finding similar binaries (default: 0.7)
- `--bert-model` - BERT model to use (default: bert-base-uncased)
- `--device` - Device to run on: cpu or cuda (default: auto-detect)

## Pipeline Components

### 1. Angr Disassembly (`angr_disassembly.py`)

Handles binary disassembly using the angr framework:
- Loads binary files
- Generates Control Flow Graphs (CFG)
- Extracts functions and basic blocks
- Retrieves assembly instructions

### 2. BERT Encoder (`bert_encoder.py`)

Encodes binary code features using BERT:
- Tokenizes assembly instructions
- Generates dense embeddings using pretrained BERT
- Processes instructions, basic blocks, and functions
- Supports batch processing for efficiency

### 3. GNN Model (`gnn_model.py`)

Graph Neural Network for processing control flow graphs:
- Implements GCN (Graph Convolutional Network) and GAT (Graph Attention Network)
- Processes graph structure information
- Enriches BERT embeddings with graph context
- Computes similarity between binary representations

### 4. Pipeline Orchestration (`pipeline.py`)

Main pipeline that ties everything together:
- Orchestrates the complete workflow
- Handles single binary and batch processing
- Computes similarity metrics between binaries
- Exports results in JSON format

## API Usage

You can also use the pipeline programmatically:

```python
from pipeline import BCSDPipeline

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