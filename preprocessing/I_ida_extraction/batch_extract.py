#!/usr/bin/env python3
"""Run the IDAPython CFG extractor over a directory of binaries."""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IDA_SCRIPT = SCRIPT_DIR / "cfg_extractor.py"

IGNORE_EXTENSIONS = {
    ".json",
    ".txt",
    ".md",
    ".csv",
    ".py",
    ".sh",
    ".c",
    ".cpp",
    ".h",
    ".o",
    ".obj",
    ".idb",
    ".i64",
    ".nam",
    ".til",
    ".id0",
    ".id1",
    ".id2",
}


def _validate_cfg_output(output_path: Path) -> None:
    """Raise if an extractor result is not a CFG JSON object."""
    with output_path.open(encoding="utf-8") as output_file:
        data = json.load(output_file)

    if not isinstance(data, dict) or not isinstance(data.get("functions"), list):
        raise ValueError("output must be a JSON object with a functions list")


def _extract_one(
    binary_path: Path,
    output_path: Path,
    ida_path: str,
    ida_script: Path,
    timeout: int,
) -> bool:
    """Extract one binary in an isolated working directory."""
    with tempfile.TemporaryDirectory(prefix="bcsd-ida-") as work_dir_name:
        work_dir = Path(work_dir_name)
        temporary_output = work_dir / "cfg.json"
        script_command = shlex.join((str(ida_script), str(temporary_output)))
        command = [
            ida_path,
            "-A",
            f"-L{work_dir / 'ida.log'}",
            f"-o{work_dir / 'database.i64'}",
            f"-S{script_command}",
            str(binary_path.resolve()),
        ]

        try:
            result = subprocess.run(
                command,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"  x Timeout ({timeout}s)")
            return False
        except OSError as error:
            print(f"  x Could not start IDA: {error}")
            return False

        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            suffix = f": {detail[-500:]}" if detail else ""
            print(f"  x IDA exited with status {result.returncode}{suffix}")
            return False

        try:
            _validate_cfg_output(temporary_output)
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as error:
            print(f"  x Invalid extractor output: {error}")
            return False

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                delete=False,
            ) as destination:
                staged_output = Path(destination.name)
                with temporary_output.open("rb") as source:
                    shutil.copyfileobj(source, destination)
            staged_output.replace(output_path)
        except OSError as error:
            if "staged_output" in locals():
                staged_output.unlink(missing_ok=True)
            print(f"  x Could not publish extractor output: {error}")
            return False
        return True


def batch_extract(
    input_dir: str | Path,
    output_dir: str | Path,
    ida_path: str = "ida64",
    ida_script: str | Path | None = None,
    timeout: int = 300,
) -> int:
    """Extract all binaries recursively and return the number that failed."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    ida_script = Path(ida_script) if ida_script else DEFAULT_IDA_SCRIPT

    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not ida_script.is_file():
        raise ValueError(f"IDA script does not exist: {ida_script}")

    files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() not in IGNORE_EXTENSIONS
        and not path.name.startswith(".")
    )

    print(f"Processing {len(files)} files from {input_dir}")
    print(f"Output directory: {output_dir}\n")

    successful = 0
    failed = 0
    for index, binary_path in enumerate(files, 1):
        relative_path = binary_path.relative_to(input_dir)
        output_path = output_dir / relative_path.parent / f"{binary_path.name}_cfg.json"
        print(f"[{index}/{len(files)}] {relative_path}")

        if _extract_one(binary_path, output_path, ida_path, ida_script, timeout):
            print(f"  -> {output_path.name}")
            successful += 1
        else:
            failed += 1

    print(f"\nDone. {successful} succeeded, {failed} failed. Output in {output_dir}")
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch IDA CFG extraction")
    parser.add_argument("input_dir", help="Directory containing binaries (searched recursively)")
    parser.add_argument("output_dir", help="Output directory for CFG JSON files")
    parser.add_argument("--ida-path", default="ida64", help="Path to the IDA executable")
    parser.add_argument("--ida-script", help="IDAPython extractor script")
    parser.add_argument("--timeout", type=int, default=300, help="Per-binary timeout in seconds")
    args = parser.parse_args(argv)

    try:
        return min(
            batch_extract(
                args.input_dir,
                args.output_dir,
                args.ida_path,
                args.ida_script,
                args.timeout,
            ),
            1,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
