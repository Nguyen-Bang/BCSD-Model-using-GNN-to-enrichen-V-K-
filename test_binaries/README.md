# Test Binaries for BCSD Model

This directory contains test binaries for validating the Graph-Aware Language Model pipeline.

## Files

- `test_gnn.c` - Source code with loops and branches for CFG testing
- `compile.sh` - Compilation script (for WSL/Linux)
- `test_binary` - Compiled x64 ELF executable (unstripped)

## Control Flow Features

The test binary includes:
1. **Loop (cycle)**: `calculate_sum()` function with a for-loop
2. **Branches**: `check_value()` with if/else statements
3. **Multiple functions**: For function-level graph extraction
4. **Debug symbols**: Compiled with `-g` flag (unstripped)

## Compilation

### On WSL/Linux:
```bash
cd test_binaries
chmod +x compile.sh
./compile.sh
```

### Manual compilation:
```bash
gcc -g -O0 -m64 -o test_binary test_gnn.c
```

## Verification

```bash
# Check file type
file test_binary

# Verify symbols are present
nm test_binary | grep -E "(calculate_sum|check_value|main)"

# Run the binary
./test_binary
```

## Expected CFG Structure

### calculate_sum function:
- Entry block
- Loop condition check
- Loop body (backward edge creates cycle)
- Exit block

### check_value function:
- Entry block
- First if condition
- Second if condition (else-if)
- Final else
- Exit block

### main function:
- Entry block
- Function call to calculate_sum
- If/else branch
- Conditional call to check_value
- Exit block
