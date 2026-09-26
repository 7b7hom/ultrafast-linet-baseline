"""Prove epoch-boundary resume preserves weights and validation outcomes."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_interrupted_training_matches_continuous_run(tmp_path):
    data, manifests = tmp_path / "data", tmp_path / "manifests"
    data.mkdir()
    manifests.mkdir()
    rng = np.random.default_rng(13)
    rows = []
    for i in range(3):
        row = {"name": f"{i}.png"}
        for kind, upper in (("low", 30), ("high", 220)):
            filename = f"{kind}_{i}.png"
            pixels = rng.integers(0, upper, (180, 180, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(data / filename)
            row[kind] = filename
            row[f"{kind}_sha256"] = hashlib.sha256((data / filename).read_bytes()).hexdigest()
        rows.append(row)
    for split, subset in (("train", rows[:2]), ("validation", rows[2:])):
        (manifests / f"{split}.json").write_text(json.dumps({"split": split, "pairs": subset}))
    cfg = json.loads((ROOT / "configs/baseline_a.json").read_text())
    cfg.update(epochs=2, batch_size=2, lr_step=1)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    base = [sys.executable, str(ROOT / "scripts/train_a.py"), "--data-root", str(data),
            "--manifests", str(manifests), "--config", str(config), "--pilot"]
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    subprocess.run(base + ["--output", str(continuous)], check=True, capture_output=True, text=True)
    subprocess.run(base + ["--output", str(resumed), "--stop-after-epoch", "1"],
                   check=True, capture_output=True, text=True)
    paused = json.loads((resumed / "summary.json").read_text())
    assert paused["completed_epochs"] == 1 and not paused["full_baseline_complete"]
    subprocess.run(base + ["--output", str(resumed), "--resume", str(resumed / "last.pt")],
                   check=True, capture_output=True, text=True)
    for name in ("last", "best"):
        a = torch.load(continuous / f"{name}.pt", weights_only=True)
        b = torch.load(resumed / f"{name}.pt", weights_only=True)
        assert a["epoch"] == b["epoch"] and a["best_psnr"] == b["best_psnr"]
        assert all(torch.equal(a["weight"][k], b["weight"][k]) for k in a["weight"])
    logs = []
    for directory in (continuous, resumed):
        with open(directory / "log.csv") as f:
            logs.append(list(csv.DictReader(f)))
    for a, b in zip(*logs):
        for key in ("epoch", "loss", "val_psnr", "val_ssim", "lr", "images_seen"):
            assert a[key] == b[key]
