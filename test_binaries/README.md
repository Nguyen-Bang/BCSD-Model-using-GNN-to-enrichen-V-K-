# Test Binaries for BCSD Model

This directory contains controlled test binaries for validating the angr disassembly module and instruction tokenization before processing the full dataset.

## Purpose

**Primary Use Cases**:
1. **Validate Disassembly Correctness**: Verify angr extracts expected CFG structures (loops, branches, function calls)
2. **Test Tokenization**: Ensure instruction sequences are tokenized correctly (opcodes separated from operands)
3. **Thesis Documentation**: Use as concrete running example in methodology chapter to illustrate:
   - How angr extracts CFG from binaries
   - How control flow patterns (loops, branches) appear in graph structure
   - How compilation variants (gcc/clang, O0/O3) affect CFG complexity
4. **Regression Testing**: Quick validation that disassembly pipeline works before full dataset processing

## Files

- `test_gnn.c` - Source code with three functions demonstrating key CFG patterns
- `compile.sh` - Compilation script creating multiple variants
- `test_gnn_gcc_O0` - Compiled with gcc -O0 (no optimization, preserves structure)
- `test_gnn_gcc_O3` - Compiled with gcc -O3 (full optimization, may inline/reorder)
- `test_gnn_clang_O0` - Compiled with clang -O0 (alternative compiler)
- `test_gnn_clang_O3` - Compiled with clang -O3 (alternative compiler, optimized)

## Control Flow Features in test_gnn.c

The test binary includes three functions with distinct CFG patterns:

### 1. `calculate_sum(int limit)` - Loop with Backward Edge
- **Pattern**: for-loop creating cycle in CFG
- **Expected CFG**: Backward edge from loop body to condition check
- **Graph Structure**:
  ```
  Entry → Loop Condition ⇄ Loop Body → Exit
               ↑______________|
           (backward edge = cycle)
  ```
- **Purpose**: Test cycle detection in CFG

### 2. `check_value(int value)` - Multi-way Branches
- **Pattern**: if/else-if/else creating multiple paths
- **Expected CFG**: Multiple outgoing edges from conditional blocks
- **Graph Structure**:
  ```
  Entry → Condition 1 → Branch A → Exit
            ↓
         Condition 2 → Branch B → Exit
            ↓
         Branch C → Exit
  ```
- **Purpose**: Test branch detection and path diversity

### 3. `main()` - Function Calls and Conditional Execution
- **Pattern**: Calls to calculate_sum and check_value with conditional logic
- **Expected CFG**: Function call nodes and conditional branches
- **Purpose**: Test inter-procedural structure and call graph extraction

## Compilation

### Using compile.sh (Recommended):
```bash
cd test_binaries
chmod +x compile.sh
./compile.sh
```

This creates four compilation variants:
- `test_gnn_gcc_O0` - gcc without optimization (baseline, most readable CFG)
- `test_gnn_gcc_O3` - gcc full optimization (may inline, loop unroll)
- `test_gnn_clang_O0` - clang without optimization (compare to gcc)
- `test_gnn_clang_O3` - clang full optimization (observe different optimizations)

### Manual compilation:
```bash
# Basic unoptimized build
gcc -g -O0 -m64 -o test_gnn_gcc_O0 test_gnn.c

# Optimized build
gcc -g -O3 -m64 -o test_gnn_gcc_O3 test_gnn.c
```

## Verification

```bash
# Check binary type
file test_gnn_gcc_O0

# Verify debug symbols are present
nm test_gnn_gcc_O0 | grep -E "(calculate_sum|check_value|main)"

# Run binary to verify functionality
./test_gnn_gcc_O0
```

**Expected output**:
```
=== GNN Test Binary ===
Calculating sum from 1 to 10...
Sum: 55
Result is large!
Category: HIGH
Done!
```

## Disassembly Testing Workflow

See quickstart.md for complete testing workflow using these binaries with the angr disassembly module.

## Using for Thesis Documentation

The test_gnn binary serves as a concrete running example in the thesis methodology chapter:
- Illustrates how angr extracts CFG from binaries
- Shows control flow patterns (loops, branches) in graph structure
- Demonstrates compilation variant effects (gcc/clang, O0/O3)
- Provides validation that disassembly pipeline works correctly
- Entry block
- Function call to calculate_sum
- If/else branch
- Conditional call to check_value
- Exit block
