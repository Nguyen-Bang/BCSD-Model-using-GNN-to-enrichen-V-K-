from dataclasses import replace
from pathlib import Path

from evaluation.binarycorp.runner import EvaluationConfig, evaluation_provenance
from evaluation.provenance import sha256_file


def _args(tmp_path: Path, model: str) -> EvaluationConfig:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    config.write_text("model: baseline\n")
    checkpoint.write_bytes(b"checkpoint")
    return EvaluationConfig(
        model=model,
        checkpoint=checkpoint,
        model_config=config,
        data_dir=tmp_path,
        split="test",
        pool_size=10_000,
        seed=0,
        batch_size=16,
        query_block_size=512,
        device="cpu",
        pretrained="hustcw/clap-asm",
        revision="revision",
        local_files_only=True,
    )


def test_binarycorp_provenance_hashes_trained_model_inputs(tmp_path, monkeypatch):
    args = _args(tmp_path, "typed-gnn")
    monkeypatch.setattr("evaluation.binarycorp.runner.git_commit", lambda: "git-revision")
    monkeypatch.setattr("evaluation.binarycorp.runner.git_dirty", lambda: True)

    result = evaluation_provenance(args)

    assert result == {
        "model_config_sha256": sha256_file(args.model_config),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "git_commit": "git-revision",
        "git_dirty": True,
        "pretrained": None,
        "revision": None,
    }


def test_binarycorp_provenance_records_zero_shot_clap_revision(tmp_path, monkeypatch):
    args = _args(tmp_path, "clap-zero-shot")
    args = replace(args, checkpoint=None, model_config=None)
    monkeypatch.setattr("evaluation.binarycorp.runner.git_commit", lambda: "git-revision")
    monkeypatch.setattr("evaluation.binarycorp.runner.git_dirty", lambda: False)

    result = evaluation_provenance(args)

    assert result["pretrained"] == "hustcw/clap-asm"
    assert result["revision"] == "revision"
    assert result["model_config_sha256"] is None
    assert result["checkpoint_sha256"] is None
