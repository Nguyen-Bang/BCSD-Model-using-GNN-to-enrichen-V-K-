import csv
from pathlib import Path

import pytest
import torch

from preprocessing.binarycorp import build_binarycorp_dataset as dataset_builder
from preprocessing.binarycorp.build_bc3m_split import (
    build_split,
    validation_projects,
)
from preprocessing.binarycorp.build_bc3m_split import (
    main as split_main,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["", "proj", "filename", "opt"])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"": str(index), **row})


def test_split_map_and_job_collection(tmp_path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    _write_csv(train_csv, [{"proj": "p1", "filename": "a-O0", "opt": "O0"}])
    _write_csv(test_csv, [{"proj": "p2", "filename": "b-O3", "opt": "O3"}])
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "a-O0_cfg.json").write_text("{}")
    (cfg_dir / "unknown_cfg.json").write_text("{}")

    split_map = dataset_builder.load_split_map(train_csv, test_csv)
    jobs, unmatched = dataset_builder.collect_jobs(cfg_dir, split_map, tmp_path / "out")

    assert unmatched == 1
    assert len(jobs) == 1
    assert jobs[0].optimization == "O0"
    assert jobs[0].output_path == tmp_path / "out" / "train" / "p1" / "a-O0.pt"


def test_process_job_writes_compact_schema(tmp_path, monkeypatch):
    cfg = tmp_path / "sample_cfg.json"
    cfg.write_text('{"binary_name":"sample","functions":[{"function_name":"f"}]}')
    output = tmp_path / "out" / "train" / "project" / "sample.pt"
    job = dataset_builder.BuildJob(cfg, output, "project", "O2")

    class FakeTokenizer:
        max_seq_length = 16

    monkeypatch.setattr(dataset_builder, "_tokenizer", FakeTokenizer())
    monkeypatch.setattr(dataset_builder, "enrich_cfg_data", lambda data: data)

    def fake_tokenize(data, _tokenizer):
        data["functions"][0].update(token_ids=[3, 4], attention_mask=[1, 1], token_type_ids=[5, 5])

    monkeypatch.setattr(dataset_builder, "tokenize_cfg_data", fake_tokenize)
    monkeypatch.setattr(
        dataset_builder,
        "reduce_function",
        lambda function: {
            "function_name": function["function_name"],
            "token_ids": torch.tensor([3]),
        },
    )

    result = dataset_builder.process_job(job)
    payload = torch.load(output, weights_only=True)

    assert result.error is None
    assert result.functions == 1
    assert payload["project"] == "project"
    assert payload["opt"] == "O2"
    assert payload["functions"][0]["function_name"] == "f"


def test_build_dataset_refuses_to_overwrite_existing_artifacts(tmp_path):
    output = tmp_path / "sample.pt"
    output.touch()
    job = dataset_builder.BuildJob(tmp_path / "sample.json", output, "project", "O2")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        dataset_builder.build_dataset([job], workers=1, max_seq_length=16)


def test_bc3m_split_is_project_disjoint_and_copied(tmp_path):
    source = tmp_path / "source"
    train_rows = []
    for project in ("p1", "p2", "p3", "p4"):
        filename = f"{project}-O0"
        train_rows.append({"proj": project, "filename": filename, "opt": "O0"})
        path = source / "train" / project / f"{filename}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"opt": "O0", "functions": []}, path)
    test_rows = [{"proj": "test-p", "filename": "test-O3", "opt": "O3"}]
    test_path = source / "test" / "test-p" / "test-O3.pt"
    test_path.parent.mkdir(parents=True)
    torch.save({"opt": "O3", "functions": []}, test_path)
    train_csv = tmp_path / "small_train.csv"
    test_csv = tmp_path / "small_test.csv"
    _write_csv(train_csv, train_rows)
    _write_csv(test_csv, test_rows)

    output = tmp_path / "bc3m"
    stats = build_split(
        source,
        train_csv,
        test_csv,
        output,
        validation_fraction=0.25,
        seed=0,
    )

    train_projects = {path.name for path in (output / "train").iterdir()}
    validation = {path.name for path in (output / "validation").iterdir()}
    assert train_projects.isdisjoint(validation)
    assert train_projects | validation == {"p1", "p2", "p3", "p4"}
    assert len(validation) == 1
    assert stats["test"] == {"files": 1, "projects": 1}
    copied = output / "test" / "test-p" / "test-O3.pt"
    assert copied.exists() and not copied.is_symlink()


def test_validation_projects_is_stable_and_exact_size():
    projects = {f"p{i}" for i in range(20)}
    first = validation_projects(projects, 0.2, seed=7)
    second = validation_projects(set(reversed(sorted(projects))), 0.2, seed=7)
    assert first == second
    assert len(first) == 4


def test_check_test_pools_rejects_any_pair_without_a_full_pool(tmp_path):
    source = tmp_path / "source"
    train_rows = [{"proj": "train-p", "filename": "train-O0", "opt": "O0"}]
    train_path = source / "train" / "train-p" / "train-O0.pt"
    train_path.parent.mkdir(parents=True)
    torch.save({"project": "train-p", "opt": "O0", "functions": []}, train_path)

    test_rows = []
    function = {"function_name": "shared"}
    for optimization in ("O0", "O1", "O2", "O3", "Os"):
        filename = f"test-{optimization}"
        test_rows.append({"proj": "test-p", "filename": filename, "opt": optimization})
        path = source / "test" / "test-p" / f"{filename}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"project": "test-p", "opt": optimization, "functions": [function]},
            path,
        )

    train_csv = tmp_path / "small_train.csv"
    test_csv = tmp_path / "small_test.csv"
    _write_csv(train_csv, train_rows)
    _write_csv(test_csv, test_rows)

    with pytest.raises(ValueError, match="No complete pool of size 2"):
        split_main(
            [
                "--source-dir",
                str(source),
                "--small-train-csv",
                str(train_csv),
                "--small-test-csv",
                str(test_csv),
                "--output-dir",
                str(tmp_path / "output"),
                "--check-test-pools",
                "--pool-size",
                "2",
            ]
        )
