"""
Node-level statistical feature extraction.

Pure computation — no file I/O. Computes instruction-type densities,
memory access patterns, and register usage from basic block instructions.

Module: preprocessing.II_static_enrichment.node_features
"""

import re
from typing import Any

register_pattern = re.compile(
    r"\b(rax|rbx|rcx|rdx|rsi|rdi|rbp|rsp|r8|r9|r10|r11|r12|r13|r14|r15|"
    r"eax|ebx|ecx|edx|esi|edi|ebp|esp|"
    r"ax|bx|cx|dx|si|di|bp|sp|"
    r"al|bl|cl|dl|ah|bh|ch|dh|"
    r"sil|dil|bpl|spl|r8b|r9b|r10b|r11b|r12b|r13b|r14b|r15b|"
    r"r8d|r9d|r10d|r11d|r12d|r13d|r14d|r15d|"
    r"r8w|r9w|r10w|r11w|r12w|r13w|r14w|r15w)\b",
    re.IGNORECASE,
)
memory_access_pattern = re.compile(r"\[.*?\]")
immediate_pattern = re.compile(r"\b(0x[0-9A-Fa-f]+|[0-9]+)\b")

ARITHMETIC_OPS = {"add", "sub", "inc", "dec", "neg", "adc", "sbb"}
MULTIPLY_DIVIDE_OPS = {"mul", "imul", "div", "idiv"}
LOGIC_OPS = {
    "and",
    "or",
    "xor",
    "not",
    "shl",
    "shr",
    "sal",
    "sar",
    "rol",
    "ror",
    "rcl",
    "rcr",
}
COMPARISON_OPS = {"cmp", "test"}
CALL_OPS = {"call"}
MOV_OPS = {"mov", "movzx", "movsx", "movsxd", "movabs"}
LEA_OPS = {"lea"}
STACK_OPS = {"push", "pop"}


def compute_node_statistics(instructions_dict: dict[str, str]) -> dict[str, Any]:
    """
    Compute statistical features for a single basic block.

    Args:
        instructions_dict: Dict mapping instruction index to instruction string.

    Returns:
        Dictionary of statistical features (densities + counts).
    """
    if not instructions_dict:
        return {
            "instruction_count": 0,
            "arithmetic_density": 0.0,
            "multiply_divide_density": 0.0,
            "logic_density": 0.0,
            "comparison_density": 0.0,
            "call_density": 0.0,
            "mov_density": 0.0,
            "lea_density": 0.0,
            "stack_op_density": 0.0,
            "memory_access_density": 0.0,
            "constant_density": 0.0,
            "unique_register_count": 0,
        }

    instruction_count = len(instructions_dict)

    arithmetic_count = 0
    multiply_divide_count = 0
    logic_count = 0
    comparison_count = 0
    call_count = 0
    mov_count = 0
    lea_count = 0
    stack_op_count = 0
    memory_access_count = 0
    constant_count = 0
    unique_registers = set()

    for instr in instructions_dict.values():
        parts = instr.split()
        if not parts:
            continue

        mnemonic = parts[0].lower()

        if mnemonic in ARITHMETIC_OPS:
            arithmetic_count += 1
        if mnemonic in MULTIPLY_DIVIDE_OPS:
            multiply_divide_count += 1
        if mnemonic in LOGIC_OPS:
            logic_count += 1
        if mnemonic in COMPARISON_OPS:
            comparison_count += 1
        if mnemonic in CALL_OPS:
            call_count += 1
        if mnemonic in MOV_OPS:
            mov_count += 1
        if mnemonic in LEA_OPS:
            lea_count += 1
        if mnemonic in STACK_OPS:
            stack_op_count += 1

        if memory_access_pattern.search(instr) and mnemonic not in LEA_OPS:
            memory_access_count += 1

        constants = immediate_pattern.findall(instr)
        constant_count += len(constants)

        registers = register_pattern.findall(instr)
        unique_registers.update(r.lower() for r in registers)

    safe_count = max(instruction_count, 1)

    return {
        "instruction_count": instruction_count,
        "unique_register_count": len(unique_registers),
        "arithmetic_density": round(arithmetic_count / safe_count, 4),
        "multiply_divide_density": round(multiply_divide_count / safe_count, 4),
        "logic_density": round(logic_count / safe_count, 4),
        "comparison_density": round(comparison_count / safe_count, 4),
        "call_density": round(call_count / safe_count, 4),
        "mov_density": round(mov_count / safe_count, 4),
        "lea_density": round(lea_count / safe_count, 4),
        "stack_op_density": round(stack_op_count / safe_count, 4),
        "memory_access_density": round(memory_access_count / safe_count, 4),
        "constant_density": round(constant_count / safe_count, 4),
    }
