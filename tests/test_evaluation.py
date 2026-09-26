"""Final-evaluation guards use synthetic pixels and never the official test set."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate as evaluation
from common import ManifestPairs, sha256, ultrafast_linet_max, write_json
from train_a import validate


def _fake_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def completed_run(tmp_path):
    """Create metadata for the complete contract, but only tiny synthetic test PNGs."""
    run_dir, manifests, data_root = (tmp_path / name for name in ("run", "manifests", "data"))
    run_dir.mkdir()
    manifests.mkdir()
    data_root.mkdir()
    manifest_hashes = {}
    for split, count in evaluation.SPLIT_COUNTS.items():
        rows = []
        for index in range(count):
            name = f"{split}_{index}.png"
            row = {"name": name}
            for key in ("low", "high"):
                relative = f"{split}/{key}/{name}"
                row[key] = relative
                if split == "test":
                    path = data_root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pixels = np.random.default_rng(index + (100 if key == "high" else 0)).integers(
                        10 if key == "low" else 80, 50 if key == "low" else 255,
                        (24, 26, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(path)
                    row[f"{key}_sha256"] = sha256(path)
                else:
                    row[f"{key}_sha256"] = _fake_hash(relative)
            rows.append(row)
        path = manifests / f"{split}.json"
        write_json(path, {"dataset": "LOL-v1", "split": split, "pairs": rows})
        manifest_hashes[split] = sha256(path)
    write_json(manifests / "summary.json", {"manifests_sha256": manifest_hashes})
    cfg = json.loads((ROOT / "configs/baseline_a.json").read_text())
    torch.manual_seed(4)
    model = ultrafast_linet_max()
    torch.save({"weight": model.state_dict()}, run_dir / "initial.pt")
    best_per_image = [
        {"name": f"validation_{index}.png", "psnr": 22.0, "ssim": .8} for index in range(50)]
    for name, epoch in (("best", 17), ("last", 360)):
        checkpoint = {"weight": model.state_dict(), "epoch": epoch,
                      "config": cfg, "pilot": False, "best_psnr": 22.0}
        if name == "last":
            checkpoint.update(best_epoch=17, best_weight=model.state_dict(), best_per_image=best_per_image)
        torch.save(checkpoint, run_dir / f"{name}.pt")
    write_json(run_dir / "run.json", {
        "config": cfg, "pilot": False, "parameters": 180,
        "train_count": 435, "validation_count": 50,
        "official_test_used": False, "initial_file_sha256": sha256(run_dir / "initial.pt"),
        "manifests_sha256": {key: manifest_hashes[key] for key in ("train", "validation")},
        "selection": "max mean per-image validation PSNR; ties keep earlier epoch",
    })
    write_json(run_dir / "summary.json", {
        "completed_epochs": 360, "planned_epochs": 360, "pilot": False, "full_baseline_complete": True,
        "official_test_used": False, "best_validation_psnr": 22.0, "best_epoch": 17,
        "checkpoint_sha256": {key: sha256(run_dir / f"{key}.pt") for key in ("best", "last")},
    })
    with (run_dir / "log.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "val_psnr", "images_seen"])
        writer.writeheader()
        for epoch in range(1, 361):
            writer.writerow({"epoch": epoch, "val_psnr": 22.0 if epoch in (17, 18) else 21.0,
                             "images_seen": 435})
    write_json(run_dir / "best_validation_per_image.json", best_per_image)
    return {"run_dir": run_dir, "manifests": manifests, "data_root": data_root,
            "output": tmp_path / "evaluation"}


def _modify_json(path, update):
    data = json.loads(path.read_text())
    update(data)
    write_json(path, data)


def _seal_checkpoint(run_dir, name, checkpoint):
    path = run_dir / f"{name}.pt"
    torch.save(checkpoint, path)
    _modify_json(run_dir / "summary.json", lambda summary: summary["checkpoint_sha256"].update({name: sha256(path)}))


def test_final_metrics_match_training_validation_protocol(completed_run):
    report = evaluation.evaluate(**completed_run, save_images=True)
    model, _, _ = evaluation.validate_completed_run(completed_run["run_dir"])
    data = ManifestPairs(completed_run["manifests"] / "test.json", completed_run["data_root"])
    reference, reference_rows = validate(model, DataLoader(data, batch_size=1), torch.device("cpu"))
    assert report["aggregate"] == reference
    assert [(row["name"], row["psnr"], row["ssim"]) for row in report["per_image"]] == [
        (row["name"], row["psnr"], row["ssim"]) for row in reference_rows]
    assert report["count"] == 15
    assert report["verification"]["selected_epoch"] == 17
    assert report["test_used_for_checkpoint_selection"] is False
    assert report["raw_dataset_images_copied"] is False
    assert len(list((completed_run["output"] / "enhanced").glob("*.png"))) == 15
    assert {path.name for path in completed_run["output"].iterdir()} == {
        "metrics.json", "per_image.csv", "aggregate.csv", "enhanced"}
    with (completed_run["output"] / "per_image.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 15
    saved = json.loads((completed_run["output"] / "metrics.json").read_text())
    assert saved["aggregate"] == reference
    with pytest.raises(FileExistsError, match="immutable"):
        evaluation.evaluate(**completed_run)


@pytest.mark.parametrize("pilot,epochs,last_epoch", [(True, 360, 360), (False, 1, 1), (False, 360, 359)])
def test_incomplete_runs_rejected_before_test_pixels(completed_run, monkeypatch, pilot, epochs, last_epoch):
    run_dir = completed_run["run_dir"]
    _modify_json(run_dir / "summary.json", lambda summary: summary.update(pilot=pilot, completed_epochs=epochs))
    last = torch.load(run_dir / "last.pt", weights_only=True)
    last["epoch"] = last_epoch
    _seal_checkpoint(run_dir, "last", last)
    monkeypatch.setattr(evaluation, "verify_test_files", lambda *args: pytest.fail("Test files accessed before completion guard"))
    with pytest.raises(ValueError, match="360"):
        evaluation.evaluate(**completed_run)
    assert not completed_run["output"].exists()


def test_checkpoint_hash_tampering_rejected(completed_run):
    with (completed_run["run_dir"] / "best.pt").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        evaluation.evaluate(**completed_run)


def test_missing_completion_summary_never_accesses_test_files(completed_run, monkeypatch):
    (completed_run["run_dir"] / "summary.json").unlink()
    monkeypatch.setattr(evaluation, "verify_test_files", lambda *args: pytest.fail("Test files accessed before completion summary"))
    with pytest.raises(ValueError, match="completed.*360-epoch.*summary"):
        evaluation.evaluate(**completed_run)
    assert not completed_run["output"].exists()


@pytest.mark.parametrize("field", ["best_epoch", "best_weight", "best_per_image"])
def test_last_checkpoint_must_agree_with_selected_best(completed_run, field):
    run_dir = completed_run["run_dir"]
    last = torch.load(run_dir / "last.pt", weights_only=True)
    if field == "best_epoch":
        last[field] = 18
    elif field == "best_weight":
        name = next(iter(last[field]))
        last[field][name] = last[field][name] + 0.01
    else:
        last[field][0]["psnr"] += 1.0
    _seal_checkpoint(run_dir, "last", last)
    with pytest.raises(ValueError, match="[Ll]ast checkpoint"):
        evaluation.evaluate(**completed_run)


def test_validation_tie_keeps_earliest_checkpoint(completed_run):
    run_dir = completed_run["run_dir"]
    best = torch.load(run_dir / "best.pt", weights_only=True)
    best["epoch"] = 18  # Same max score, but wrong tie-breaking checkpoint.
    _seal_checkpoint(run_dir, "best", best)
    with pytest.raises(ValueError, match="earliest maximum validation"):
        evaluation.evaluate(**completed_run)


def test_checkpoint_configuration_cannot_change(completed_run):
    run_dir = completed_run["run_dir"]
    best = torch.load(run_dir / "best.pt", weights_only=True)
    best["config"]["selection"] = "test_psnr"
    _seal_checkpoint(run_dir, "best", best)
    with pytest.raises(ValueError, match="checkpoint metadata"):
        evaluation.evaluate(**completed_run)


def test_manifest_change_rejected_even_if_pixels_unchanged(completed_run):
    path = completed_run["manifests"] / "test.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest SHA256"):
        evaluation.evaluate(**completed_run)


def test_split_leakage_detected_even_with_updated_manifest_seal(completed_run):
    manifests = completed_run["manifests"]
    train = json.loads((manifests / "train.json").read_text())
    _modify_json(manifests / "test.json", lambda test: test["pairs"][0].update(
        high_sha256=train["pairs"][0]["high_sha256"]))
    _modify_json(manifests / "summary.json", lambda summary: summary["manifests_sha256"].update(
        test=sha256(manifests / "test.json")))
    with pytest.raises(ValueError, match="Split overlap.*high_sha256"):
        evaluation.evaluate(**completed_run)


def test_changed_test_image_rejected_before_decoding(completed_run, monkeypatch):
    test = json.loads((completed_run["manifests"] / "test.json").read_text())
    (completed_run["data_root"] / test["pairs"][0]["low"]).write_bytes(b"invalid PNG")
    monkeypatch.setattr(evaluation, "ManifestPairs", lambda *args: pytest.fail("Decoded before checking hashes"))
    with pytest.raises(ValueError, match="Test image SHA256 mismatch"):
        evaluation.evaluate(**completed_run)


def test_training_log_must_cover_all_epochs(completed_run):
    path = completed_run["run_dir"] / "log.csv"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="each completed epoch"):
        evaluation.evaluate(**completed_run)
