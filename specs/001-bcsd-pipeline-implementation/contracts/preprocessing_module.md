# Preprocessing Module Contract (PHASE 1)

**Modules**: `preprocessing.extract_features` + `preprocessing.tokenizer`  
**Purpose**: Extract CFGs from binaries (angr) + Custom assembly tokenization  
**Owner**: User Story 1 (P1)

---

## Module Overview

**PHASE 1** has two critical components:

1. **Feature Extraction** (`preprocessing/extract_features.py`):
   - Uses angr CFGFast to extract Control Flow Graph structure
   - Outputs CFG edge lists + basic block addresses
   - Outputs raw disassembled instruction strings

2. **Custom Tokenization** (`preprocessing/tokenizer.py`):
   - Builds domain-specific vocabulary from assembly opcodes and operands
   - Tokenizes instructions (NOT using BERT's tokenizer)
   - Handles special tokens ([PAD], [MASK], [CLS], [SEP])
   - Outputs token IDs ready for BERT input

---

## Public API: Feature Extraction

**Module**: `preprocessing.extract_features`

### Function: `extract_cfg(binary_path: str, output_dir: str, timeout: int = 600) -> Dict[str, Any]`

**Description**: Extracts CFG from an ELF binary using angr's CFGFast.

**Parameters**:
- `binary_path` (str): Absolute path to the binary executable
- `output_dir` (str): Directory to save output JSON files
- `timeout` (int, optional): Maximum analysis time in seconds (default: 600)

**Returns**: Dictionary with structure:
```python
{
    "status": "success" | "failed",
    "file_hash": "sha256_hash_string",
    "output_file": "/path/to/data/preprocessed/{hash}.json",  # only if success (SINGLE unified file)
    "error": "error message",  # only if failed
    "node_count": int,  # only if success
    "edge_count": int,  # only if success
    "token_count": int,  # only if success
    "processing_time_seconds": float
}
```

**Side Effects**:
- Creates ONE JSON file in `data/preprocessed/`: `{file_hash}.json` containing:
  ```json
  {
    "id": "a1b2c3d4...",
    "function_name": "ssl_read",
    "tokens": ["MOV", "EAX", "EBX", ...],
    "token_count": 245,
    "edges": [[0, 1], [1, 2], ...],
    "node_count": 128,
    "edge_count": 156
  }
  ```
- Appends entry to `data/metadata.csv` with columns: file_hash, project, function_name, optimization, compiler, split_set
- Logs warnings/errors to console

**Error Handling**:
- File not found → raises `FileNotFoundError`
- Timeout exceeded → returns `{"status": "failed", "error": "Timeout"}`
- angr analysis error → returns `{"status": "failed", "error": <message>}`
- Corrupted binary → returns `{"status": "failed", "error": "Parse error"}`

**Performance**:
- Expected: <5 minutes for binaries <10MB
- Scales linearly with binary size

---

## Public API: Custom Tokenization

**Module**: `preprocessing.tokenizer`

### Class: `AssemblyTokenizer`

**Description**: Domain-specific tokenizer for assembly instructions. Builds vocabulary from opcodes, registers, immediates. NOT BERT's WordPiece tokenizer.

**Methods**:

#### `__init__(self, vocab_size: int = 5000, max_seq_length: int = 512)`

**Parameters**:
- `vocab_size` (int): Maximum vocabulary size (default: 5000, configurable for experimentation)
- `max_seq_length` (int): Maximum sequence length (for padding/truncation)

#### `build_vocab(self, disasm_files: List[str]) -> None`

**Description**: Builds vocabulary from collection of disassembly JSON files.

**Parameters**:
- `disasm_files` (List[str]): Paths to disassembly JSON files

**Side Effects**:
- Populates internal vocabulary mapping
- Assigns token IDs to opcodes, registers, immediates
- Reserves special tokens: [PAD]=0, [CLS]=101, [SEP]=102, [MASK]=103, [UNK]=104

#### `tokenize(self, disasm_file: str, output_file: str) -> Dict[str, Any]`

**Description**: Tokenizes disassembly instructions and saves to JSON.

**Parameters**:
- `disasm_file` (str): Path to disassembly JSON file
- `output_file` (str): Path to save tokenized output

**Returns**:
```python
{
    "status": "success" | "failed",
    "token_file": "/path/to/output_tokens.json",
    "vocab_size": int,
    "sequence_length": int,
    "error": "error message"  # only if failed
}
```

**Tokenization Strategy**:
- Split each instruction: `"mov rax, rdi"` → `["mov", "rax", "rdi"]`
- Map to token IDs from vocabulary
- Add special tokens: [CLS] at start, [SEP] at end
- Pad to `max_seq_length` with [PAD]
- Generate attention mask (1 for real tokens, 0 for padding)

#### `save_vocab(self, vocab_file: str) -> None`

**Description**: Saves vocabulary to JSON file for reproducibility.

#### `load_vocab(self, vocab_file: str) -> None`

**Description**: Loads vocabulary from JSON file.

---

## Public API: Batch Processing

### Function: `preprocess_dataset(binary_dir: str, output_dir: str, vocab_file: str = None) -> Dict[str, Any]`

**Description**: Batch processes all binaries in a directory (extract CFG + tokenize).

**Parameters**:
- `binary_dir` (str): Directory containing binary executables
- `output_dir` (str): Directory to save outputs
- `vocab_file` (str, optional): Path to existing vocab JSON (if None, builds new vocab)

**Returns**:
```python
{
    "total_binaries": int,
    "successful": int,
    "failed": int,
    "output_files": List[str],  # List of {hash}.json files in data/preprocessed/
    "metadata_file": "data/metadata.csv",
    "vocab_file": str,
    "processing_time_seconds": float
}
```

**Workflow**:
1. Extract CFGs + tokenize all binaries simultaneously (produces one {hash}.json per function)
2. If no vocab_file provided, build vocab from all instruction sequences
3. Write all entries to `data/metadata.csv` with schema: file_hash, project, function_name, optimization, compiler, split_set
4. Save vocab to `{output_dir}/vocab.json`

**Output Structure**:
```
data/
├── metadata.csv              # Master index
├── preprocessed/
│   ├── a1b2c3d4.json         # Single merged file per function
│   ├── e5f6g7h8.json
│   └── ...
└── vocab.json                # Tokenizer vocabulary
```

---

## Legacy API (Deprecated)

### Function: `load_cfg(cfg_file_path: str) -> ControlFlowGraph`

**Status**: DEPRECATED - Use `preprocessing.extract_features.load_cfg()` instead

**Description**: Loads a previously extracted CFG from JSON file.

**Parameters**:
- `cfg_file_path` (str): Path to CFG JSON file

**Returns**: `ControlFlowGraph` object (as defined in data-model.md)

**Error Handling**:
- File not found → raises `FileNotFoundError`
- Invalid JSON → raises `ValueError`
- Schema mismatch → raises `ValidationError`

---

### Function: `load_instruction_sequence(seq_file_path: str) -> InstructionSequence`

**Status**: DEPRECATED - Use `preprocessing.tokenizer.AssemblyTokenizer.load_tokens()` instead

**Description**: Loads tokenized instruction sequence from JSON file.

**Parameters**:
- `seq_file_path` (str): Path to sequence JSON file

**Returns**: `InstructionSequence` object (as defined in data-model.md)

**Error Handling**:
- Same as `load_cfg`

---

## Output Schema

### CFG JSON File (`{binary_hash}_cfg.json`)

```json
{
    "binary_hash": "sha256_hex_string",
    "binary_name": "example_binary",
    "architecture": "x86_64",
    "is_stripped": false,
    "extracted_at": "2025-12-13T10:30:00",
    "node_count": 150,
    "edge_count": 200,
    "entry_points": [0, 5],
    "nodes": [
        {
            "node_id": 0,
            "start_address": "0x401000",
            "end_address": "0x401020",
            "instruction_count": 8,
            "instructions": ["mov rax, rdi", "call 0x402000", "..."],
            "function_name": "main",
            "block_type": "entry"
        },
        {
            "node_id": 1,
            "start_address": "0x401030",
            "end_address": "0x401050",
            "instruction_count": 5,
            "instructions": ["test eax, eax", "jne 0x401060", "..."],
            "function_name": "main",
            "block_type": "branch"
        }
    ],
    "edges": [
        [0, 1],
        [1, 2],
        [1, 3]
    ],
    "metadata": {
        "angr_version": "9.2.0",
        "cfg_method": "CFGFast",
        "auto_load_libs": false
    }
}
```

### Tokenized Sequence JSON File (`{binary_hash}_tokens.json`)

**Note**: Output from `preprocessing/tokenizer.py` (NOT BERT's tokenizer)

```json
{
    "binary_hash": "sha256_hex_string",
    "vocab_size": 5000,
    "max_seq_length": 512,
    "token_ids": [101, 234, 567, 890, 1234, ..., 102],
    "attention_mask": [1, 1, 1, 1, 1, ..., 0, 0],
    "special_tokens": {
        "[PAD]": 0,
        "[CLS]": 101,
        "[SEP]": 102,
        "[MASK]": 103,
        "[UNK]": 104
    },
    "vocab_sample": {
        "mov": 234,
        "rax": 567,
        "rdi": 890,
        "call": 1234,
        "0x402000": 1235
    },
    "metadata": {
        "created_at": "2025-12-13T10:30:00",
        "total_instructions": 256,
        "tokenization_strategy": "opcode_operand_split"
    }
}
```

**Tokenization Strategy**: 
- Split instructions into [opcode, operand1, operand2, ...]
- Build vocab from unique opcodes, registers, immediate values
- NOT using BERT's WordPiece tokenizer (assembly-specific)
```

---

## Dependencies

**Required Libraries**:
- angr >= 9.2.0
- networkx >= 2.8
- json (stdlib)
- hashlib (stdlib)

**Configuration**:
- Environment variable `ANGR_TIMEOUT` (overrides default timeout)
- Logging configured via Python's logging module

---

## Testing Interface

**Test Binary Validation** (Primary Development Test):
- **Test File**: `test_binaries/test_gnn` (compiled from test_gnn.c)
- **Purpose**: Validate disassembly correctness and tokenization before processing full dataset
- **Validation Steps**:
  1. Compile test_gnn.c with multiple variants using `test_binaries/compile.sh`:
     - `test_gnn_gcc_O0` (gcc without optimization)
     - `test_gnn_gcc_O3` (gcc full optimization)
     - `test_gnn_clang_O0` (clang without optimization)
     - `test_gnn_clang_O3` (clang full optimization)
  2. Run `extract_function_data()` on each variant
  3. Manually inspect output JSON to verify:
     - **calculate_sum function**: Shows loop cycle (backward edge [2, 1] from loop body to condition)
     - **check_value function**: Shows if/else branches (multiple outgoing edges from conditional blocks)
     - **main function**: Shows function calls and conditional paths
     - **Instruction tokenization**: Opcodes and operands correctly separated (e.g., "mov", "rax", "rdi")
  4. Compare CFG structures across compilation variants to observe optimization effects
  5. Use validated output as concrete example in thesis methodology chapter
- **Expected Output Structure**:
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
        "edges": [[0, 1], [1, 2], [2, 1], [1, 3]]  // [2,1] = loop backward edge
      },
      {"function_name": "check_value", "blocks": [...], "edges": [[0,1], [0,2], ...]},
      {"function_name": "main", "blocks": [...], "edges": [...]}
    ]
  }
  ```

**Unit Tests**:
- `test_extract_cfg_test_binary()`: Run on test_binaries/test_gnn_gcc_O0, verify 3 functions extracted
- `test_loop_detection()`: Verify calculate_sum shows backward edge indicating loop
- `test_branch_detection()`: Verify check_value shows multiple paths from if/else
- `test_compilation_variants()`: Compare gcc vs clang, O0 vs O3 CFG differences
- `test_instruction_tokenization()`: Verify opcodes/operands separated correctly
- `test_extract_cfg_timeout()`: Large binary with short timeout → returns timeout error
- `test_extract_cfg_corrupted()`: Invalid binary → returns parse error
- `test_load_cfg_valid()`: Valid JSON → returns ControlFlowGraph object

**Integration Test**:
- Run on test_binaries first, validate manually
- Then run on small binary from Dataset-1 (e.g., curl)
- Verify JSON files are created
- Load CFG and verify node/edge counts match expected values

---

## Example Usage

```python
from pipeline.angr_disassembly import extract_cfg, load_cfg

# Extract CFG from binary
result = extract_cfg(
    binary_path="/path/to/Dataset-1/clamav/clamscan",
    output_dir="data/preprocessed",
    timeout=300
)

if result["status"] == "success":
    print(f"Extracted {result['node_count']} nodes, {result['edge_count']} edges")
    
    # Load the CFG
    cfg = load_cfg(result["cfg_file"])
    print(f"Loaded CFG with {cfg.node_count} nodes")
else:
    print(f"Extraction failed: {result['error']}")
```

---

## Performance Benchmarks

| Binary Size | Expected Time | Max Memory |
|-------------|---------------|------------|
| <1 MB | <30 seconds | <500 MB |
| 1-5 MB | 1-3 minutes | <1 GB |
| 5-10 MB | 3-5 minutes | <2 GB |
| >10 MB | Use timeout | <4 GB |

---

## Validation Rules (Enforced at Output)

1. `binary_hash` must be valid SHA256 (64 hex chars)
2. `node_count` must match length of `nodes` array
3. `edge_count` must match length of `edges` array
4. All edge references must be valid node IDs (0 to node_count-1)
5. `entry_points` must be subset of node IDs
6. Each node must have at least 1 instruction
7. `tokens` array must not be empty
8. Token count must match length of `tokens` array
