/*
 * test_gnn.c - Minimal test binary for Graph-Aware Language Model (BCSD)
 * 
 * Simplified to contain only basic control flow to ensure a small CFG.
 */

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
 * Main function - entry point
 */
int main() {
    int limit = 10;
    
    // Just call the function and return the result
    // This avoids linking in stdio/printf overhead
    int result = calculate_sum(limit);
    
    return result;
}