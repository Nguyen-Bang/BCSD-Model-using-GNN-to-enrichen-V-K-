/*
 * test_gnn.c - Test binary for Graph-Aware Language Model (BCSD)
 * 
 * This program contains:
 * - A loop (for cycle detection in CFG)
 * - Branches (if/else for control flow diversity)
 * - Multiple functions (for function-level analysis)
 */

#include <stdio.h>

/**
 * Function with a loop - creates a cycle in the CFG
 */
int calculate_sum(int limit) {
    int sum = 0;
    
    // Loop creates a backward edge in the CFG (cycle)
    for (int i = 1; i <= limit; i++) {
        sum += i;
    }
    
    return sum;
}

/**
 * Function with branches - creates multiple paths in the CFG
 */
const char* check_value(int value) {
    if (value > 50) {
        return "HIGH";
    } else if (value > 20) {
        return "MEDIUM";
    } else {
        return "LOW";
    }
}

/**
 * Main function with control flow
 */
int main() {
    int limit = 10;
    
    printf("=== GNN Test Binary ===\n");
    printf("Calculating sum from 1 to %d...\n", limit);
    
    int result = calculate_sum(limit);
    printf("Sum: %d\n", result);
    
    // Branch based on result
    if (result > 50) {
        printf("Result is large!\n");
        const char* category = check_value(result);
        printf("Category: %s\n", category);
    } else {
        printf("Result is small.\n");
    }
    
    printf("Done!\n");
    return 0;
}
