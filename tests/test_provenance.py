from evaluation.provenance import provenance, sha256_file


def test_provenance_hashes_config_and_checkpoint(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.pt"
    config.write_text("model: baseline\n")
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr("evaluation.provenance.git_commit", lambda: "revision")
    monkeypatch.setattr("evaluation.provenance.git_dirty", lambda: True)

    result = provenance(config, checkpoint)

    assert result == {
        "config": str(config),
        "config_sha256": sha256_file(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "git_commit": "revision",
        "git_dirty": True,
    }
