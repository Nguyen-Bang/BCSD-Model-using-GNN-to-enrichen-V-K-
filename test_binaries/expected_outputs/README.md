# Expected CFG Outputs for Test Binaries

This directory contains annotated examples of expected CFG extraction results from test_gnn.c binaries.

## Test Binary: test_gnn.c

The test binary contains three functions demonstrating key CFG patterns:

### 1. calculate_sum(int n)
**Expected CFG Pattern**: Loop with backward edge
- **Entry block**: Initialize variables (sum=0, i=0)
- **Loop condition block**: Compare i <= n
- **Loop body block**: Add i to sum, increment i  
- **Backward edge**: From loop body back to loop condition (creates cycle)
- **Exit block**: Return sum

### 2. check_value(int x)
**Expected CFG Pattern**: Conditional branches (if/else)
- **Entry block**: Load parameter x
- **Condition block**: Compare x == 42
- **True branch**: Print "Value is 42!"
- **False branch**: Print "Value is not 42."
- **Merge block**: Return appropriate string
- **Multiple edges**: From condition to both branches

### 3. main()
**Expected CFG Pattern**: Function calls and sequential execution
- **Entry blocks**: Variable initialization
- **Call blocks**: Calls to calculate_sum(), check_value()
- **Print blocks**: Multiple printf() calls
- **Control flow**: Sequential with some conditionals

## Compilation Variants

### gcc -O0 (No optimization)
- **Node count**: ~67 nodes (as extracted)
- **Edge count**: ~86 edges
- **Characteristics**:
  - More basic blocks due to no optimization
  - Clear separation of operations
  - Easy to identify source code structure
  - Backward edge clearly visible in calculate_sum
  - Branches clearly separated in check_value

### gcc -O3 (Full optimization)
- **Node count**: Expected to be fewer (~40-50 nodes)
- **Edge count**: Expected to be fewer
- **Characteristics**:
  - Loop unrolling may modify calculate_sum structure
  - Inlining may merge function calls
  - Dead code elimination
  - Some blocks may be merged

### clang -O0 (No optimization)
- **Node count**: Similar to gcc -O0 (~65-70)
- **Edge count**: Similar to gcc -O0
- **Characteristics**:
  - Different instruction sequences but similar structure
  - Different register allocation
  - May have different prologue/epilogue

### clang -O3 (Full optimization)  
- **Node count**: Expected fewer nodes
- **Edge count**: Expected fewer edges
- **Characteristics**:
  - More aggressive optimization than gcc -O3
  - May have different optimization strategies
  - Loop vectorization possible

## Key Validation Points

When manually inspecting CFG JSON outputs, verify:

1. **calculate_sum has loop**:
   ```
   - Look for backward edge in edges list
   - Pattern: [higher_node_id, lower_node_id] indicating loop
   ```

2. **check_value has branches**:
   ```
   - Look for node with multiple outgoing edges  
   - Pattern: [condition_node, true_branch], [condition_node, false_branch]
   ```

3. **main has function calls**:
   ```
   - Look for call instructions to calculate_sum, check_value
   - Pattern: blocks containing "call 0x<address>"
   ```

4. **All functions identified**:
   ```
   - Check "functions" list in JSON
   - Should contain: calculate_sum, check_value, main
   - May also contain: _start, _init, _fini, and PLT entries
   ```

## Files

The actual extracted CFG JSON files are stored in `/data/test_output/` with filenames based on binary hash:
- Format: `{sha256_hash}_cfg.json`
- Use these files to manually verify the CFG structure matches expectations above

## Manual Validation Process (T012-T014)

1. Run extraction on test_gnn_gcc_O0
2. Open the generated JSON file
3. Search for "calculate_sum" in functions list
4. Find the corresponding basic blocks  
5. Verify backward edge exists (loop pattern)
6. Search for "check_value" in functions list
7. Verify multiple outgoing edges from condition block (branch pattern)
8. Compare O0 vs O3 variants to observe optimization effects
