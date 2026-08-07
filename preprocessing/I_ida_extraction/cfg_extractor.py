"""Extract raw control-flow graphs with IDAPython's FlowChart API.

This script performs ONLY IDA-native analysis (requires IDA Pro runtime).
Static features are added later by ``II_static_enrichment``.

Scope:
- Only functions in the .text segment are extracted. Compiler/linker
  boilerplate (.init, .fini, .plt, .extern) and FLIRT-identified library
  functions are excluded — they carry no useful signal for BCSD.

Features:
- CLAP-compatible instruction rebasing (loc_XXXXX -> INSTRN)
- Edge type classification (requires IDA API)
- Basic structural metadata (addresses, sizes, counts)
- Native IDA features: XREFs, string refs, call targets, data refs

Edge Types:
- conditional_jump: Branch instructions (je, jne, jg, etc.)
- unconditional_jump: Direct jmp instructions
- indirect_jump: Jump through register/memory
- switch: Computed jumps (jump tables)
- fallthrough: Sequential execution

Invocation:
    ida64 -A -S"cfg_extractor.py output.json" <binary>
"""

import json
import re
from pathlib import Path

try:
    import ida_funcs
    import ida_gdl
    import ida_nalt
    import ida_segment
    import ida_ua
    import idaapi
    import idautils
    import idc

    RUNNING_IN_IDA = True
except ImportError:
    RUNNING_IN_IDA = False

loc_pattern = re.compile(r" (loc|locret|sub)_(\w+)")
offset_pattern = re.compile(r"\$\+(\w+)")


def extract_basic_block_instructions(start_ea, end_ea):
    """Extract all instructions within a basic block range (IDA-native)."""
    instructions = {}
    for head in idautils.Heads(start_ea, end_ea):
        if idc.is_code(idc.get_full_flags(head)):
            disasm = idc.generate_disasm_line(head, 0)
            if disasm:
                instructions[head] = disasm.strip()
    return instructions


def rebase_instructions(instructions_dict, _func_start_ea):
    """
    Rebase instruction addresses to instruction indices (CLAP-compatible).
    Converts 'jz loc_401AA2' to 'jz INSTR3', similar to CLAP's approach.
    """
    index = 1
    rebased = {}
    addrs = sorted(instructions_dict.keys())

    for addr in addrs:
        instr = instructions_dict[addr]

        for prefix, target_addr in loc_pattern.findall(instr):
            try:
                target_instr_idx = addrs.index(int(target_addr, 16)) + 1
                instr = instr.replace(f" {prefix}_{target_addr}", f" INSTR{target_instr_idx}")
            except ValueError:
                continue

        for offset in offset_pattern.findall(instr):
            offset_val = int(offset, 16)
            target_addr = addr + offset_val
            try:
                target_instr_idx = addrs.index(target_addr) + 1
                instr = instr.replace(f"$+{offset}", f"INSTR{target_instr_idx}")
            except ValueError:
                continue

        rebased[str(index)] = instr
        index += 1

    return rebased


def extract_xrefs(start_ea, end_ea):
    """Extract cross-reference counts (IDA-native)."""
    incoming_xrefs = set()
    outgoing_xrefs = set()

    for head in idautils.Heads(start_ea, end_ea):
        if idc.is_code(idc.get_full_flags(head)):
            for xref in idautils.CodeRefsTo(head, 0):
                if xref < start_ea or xref >= end_ea:
                    incoming_xrefs.add(xref)
            for xref in idautils.CodeRefsFrom(head, 0):
                if xref < start_ea or xref >= end_ea:
                    outgoing_xrefs.add(xref)

    return len(incoming_xrefs), len(outgoing_xrefs)


def extract_string_references(start_ea, end_ea):
    """Count string references in the block (IDA-native)."""
    string_refs = set()
    for head in idautils.Heads(start_ea, end_ea):
        if idc.is_code(idc.get_full_flags(head)):
            for xref in idautils.DataRefsFrom(head):
                str_type = idc.get_str_type(xref)
                if str_type is not None and str_type >= 0:
                    string_refs.add(xref)
    return len(string_refs)


def extract_call_targets(start_ea, end_ea):
    """Extract called function names (IDA-native)."""
    call_targets = []
    for head in idautils.Heads(start_ea, end_ea):
        if idc.is_code(idc.get_full_flags(head)):
            mnem = idc.print_insn_mnem(head)
            if mnem and mnem.lower() == "call":
                for xref in idautils.CodeRefsFrom(head, 0):
                    func_name = idc.get_func_name(xref)
                    if func_name:
                        call_targets.append(func_name)
                    else:
                        call_targets.append(f"sub_{xref:X}")
    return call_targets


def extract_data_references(start_ea, end_ea):
    """Count data references (global data, constants) in the block (IDA-native)."""
    data_refs = set()
    for head in idautils.Heads(start_ea, end_ea):
        if idc.is_code(idc.get_full_flags(head)):
            for xref in idautils.DataRefsFrom(head):
                str_type = idc.get_str_type(xref)
                if str_type is None or str_type < 0:
                    data_refs.add(xref)
    return len(data_refs)


def get_edge_type(block, succ_block):
    """Classify edge type based on last instruction (IDA-native)."""
    last_ea = idc.prev_head(block.end_ea)
    if last_ea == idc.BADADDR:
        return "fallthrough"

    mnem = idc.print_insn_mnem(last_ea)
    if not mnem:
        return "fallthrough"

    mnem = mnem.lower()

    if mnem.startswith("j") and mnem != "jmp":
        target_ea = idc.get_operand_value(last_ea, 0)
        return "conditional_jump" if succ_block.start_ea == target_ea else "fallthrough"

    if mnem == "jmp":
        insn = ida_ua.insn_t()
        if ida_ua.decode_insn(insn, last_ea) > 0 and insn.ops[0].type in [
            ida_ua.o_reg,
            ida_ua.o_phrase,
            ida_ua.o_displ,
            ida_ua.o_mem,
        ]:
            if ida_nalt.get_switch_info(last_ea):
                return "switch"
            return "indirect_jump"
        return "unconditional_jump"

    return "fallthrough"


def extract_function_cfg(func_ea):
    """
    Extract raw CFG for a single function with CLAP-compatible rebasing.

    Returns only IDA-native data. Function-level statistics (node_count,
    edge_count, cyclomatic_complexity, function_size) are computed in
    post-processing by ``II_static_enrichment/function_features.py``.
    """
    func = ida_funcs.get_func(func_ea)
    if not func:
        return None

    func_name = idc.get_func_name(func_ea) or f"sub_{func_ea:X}"
    flowchart = ida_gdl.FlowChart(func)

    all_instructions = {}
    block_ranges = {}
    for block in flowchart:
        instrs = extract_basic_block_instructions(block.start_ea, block.end_ea)
        all_instructions.update(instrs)
        block_ranges[block.id] = (block.start_ea, block.end_ea)

    rebased = rebase_instructions(all_instructions, func_ea)
    addrs = sorted(all_instructions.keys())
    addr_to_idx = {addr: str(i + 1) for i, addr in enumerate(addrs)}

    nodes = []
    for block in flowchart:
        start_ea, end_ea = block_ranges[block.id]
        block_instrs = {
            addr_to_idx[a]: rebased[addr_to_idx[a]] for a in addrs if start_ea <= a < end_ea
        }

        incoming_xrefs, outgoing_xrefs = extract_xrefs(start_ea, end_ea)
        string_ref_count = extract_string_references(start_ea, end_ea)
        call_targets = extract_call_targets(start_ea, end_ea)
        data_ref_count = extract_data_references(start_ea, end_ea)

        nodes.append(
            {
                "id": block.id,
                "start_addr": hex(start_ea),
                "end_addr": hex(end_ea),
                "size": end_ea - start_ea,
                "instructions": block_instrs,
                "instruction_count": len(block_instrs),
                "successor_count": len(list(block.succs())),
                "predecessor_count": len(list(block.preds())),
                "incoming_xref_count": incoming_xrefs,
                "outgoing_xref_count": outgoing_xrefs,
                "string_ref_count": string_ref_count,
                "call_targets": call_targets,
                "call_count": len(call_targets),
                "data_ref_count": data_ref_count,
            }
        )

    edges = [
        [block.id, succ.id, get_edge_type(block, succ)]
        for block in flowchart
        for succ in block.succs()
    ]

    return {
        "function_name": func_name,
        "function_addr": hex(func_ea),
        "nodes": nodes,
        "edges": edges,
        "rebased_instructions": rebased,
    }


def extract_all_cfgs():
    """Extract CFGs for all functions in the binary."""
    idaapi.auto_wait()

    functions_data = []
    total_nodes = total_edges = 0

    for func_ea in idautils.Functions():
        func = ida_funcs.get_func(func_ea)
        seg = ida_segment.getseg(func_ea)
        seg_name = ida_segment.get_segm_name(seg) if seg else ""
        if not func or func.flags & ida_funcs.FUNC_LIB or seg_name != ".text":
            continue

        func_cfg = extract_function_cfg(func_ea)
        if func_cfg and len(func_cfg["nodes"]) > 0:
            functions_data.append(func_cfg)
            total_nodes += len(func_cfg["nodes"])
            total_edges += len(func_cfg["edges"])

    print(f"Extracted {len(functions_data)} functions, {total_nodes} nodes, {total_edges} edges")

    return {
        "binary_name": idc.get_root_filename(),
        "binary_path": ida_nalt.get_input_file_path(),
        "functions": functions_data,
        "metadata": {
            "total_functions": len(functions_data),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "ida_version": idaapi.get_kernel_version(),
        },
    }


def main():
    """Extract all CFGs to the output path passed after the script name."""
    if not RUNNING_IN_IDA:
        print("Error: Must run inside IDA Pro")
        return 1

    if len(idc.ARGV) != 2:
        print("Error: Expected an explicit output path")
        idc.qexit(2)
        return 2

    output_file = Path(idc.ARGV[1]).resolve()
    print(f"Output file: {output_file}")

    try:
        result = extract_all_cfgs()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
    except Exception as error:
        print(f"CFG extraction failed: {error}")
        idc.qexit(1)
        return 1

    print(f"Successfully saved CFG data to: {output_file}")

    idc.qexit(0)
    return 0


if __name__ == "__main__":
    if RUNNING_IN_IDA:
        main()
