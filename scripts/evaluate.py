"""One final sigma=0 test evaluation after an independently completed 360-epoch A run.

Only the checkpoint already selected by validation PSNR is loaded. Test images are
not decoded until completion, checkpoint selection, and split integrity checks pass.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import torch

from common import (ROOT, ManifestPairs, environment, psnr, save_image, seed_runtime,
                    sha256, ssim, ultrafast_linet_max, write_json)

SPLIT_COUNTS = {"train": 435, "validation": 50, "test": 15}
METRIC_PROTOCOL = {
    "color_space": "RGB",
    "dtype": "float32",
    "data_range": 1.0,
    "resolution": "full input resolution",
    "prediction_clamp": [0.0, 1.0],
    "border_crop": 0,
    "brightness_matching": False,
    "quantize_before_scoring": False,
    "aggregation": "arithmetic mean of per-image scores",
    "ssim": "pinned vendor.metrics.ssim / pytorch_msssim.ssim defaults",
    "psnr": "pinned vendor.metrics.psnr; exact equality returns 100 dB",
}


def _read_json(path):
    return json.loads(Path(path).read_text())


def _same_score(actual, expected, label):
    if not math.isfinite(float(actual)) or not math.isclose(
            float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError(f"Inconsistent {label}")


def validate_completed_run(run_dir):
    """Inspect run artifacts only; never touch dataset pixels here."""
    run_dir = Path(run_dir)
    run = _read_json(run_dir / "run.json")
    if not (run_dir / "summary.json").is_file():
        raise ValueError("Final evaluation requires a completed non-pilot 360-epoch baseline summary")
    summary = _read_json(run_dir / "summary.json")
    cfg = run["config"]
    if (run.get("pilot") is not False or summary.get("pilot") is not False
            or summary.get("full_baseline_complete") is not True
            or cfg.get("epochs") != 360 or summary.get("completed_epochs") != 360
            or summary.get("planned_epochs") != 360):
        raise ValueError("Final evaluation requires a completed non-pilot 360-epoch baseline")
    if (cfg.get("experiment") != "A" or cfg.get("additional_noise_probability") != 0
            or cfg.get("selection") != "validation_psnr_sigma0"):
        raise ValueError("Expected frozen A / validation_psnr_sigma0 selection policy")
    if run.get("official_test_used") is not False or summary.get("official_test_used") is not False:
        raise ValueError("Official test must not have been used during training or checkpoint selection")
    if run.get("parameters") != 180:
        raise ValueError("Expected the unchanged 180-parameter UltraFast-LiNET-Max")
    if sha256(run_dir / "initial.pt") != run["initial_file_sha256"]:
        raise ValueError("Initial checkpoint SHA256 mismatch")

    checkpoint_hashes = {name: sha256(run_dir / f"{name}.pt") for name in ("best", "last")}
    if summary.get("checkpoint_sha256") != checkpoint_hashes:
        raise ValueError("Completed checkpoint SHA256 mismatch or missing checkpoint seals")
    checkpoints = {
        name: torch.load(run_dir / f"{name}.pt", map_location="cpu", weights_only=True)
        for name in ("best", "last")
    }
    for name, checkpoint in checkpoints.items():
        if checkpoint.get("pilot") is not False or checkpoint.get("config") != cfg:
            raise ValueError(f"{name} checkpoint metadata does not match frozen run configuration")
        if not isinstance(checkpoint.get("weight"), dict):
            raise ValueError(f"{name} checkpoint has no state dictionary")
    if checkpoints["last"].get("epoch") != 360:
        raise ValueError("Last checkpoint must have completed epoch 360")
    with (run_dir / "log.csv").open(newline="") as handle:
        log = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in log] != list(range(1, 361)):
        raise ValueError("Training log must contain each completed epoch 1 through 360 exactly once")
    if any(int(row["images_seen"]) != SPLIT_COUNTS["train"] for row in log):
        raise ValueError("Each epoch must have processed all 435 training pairs")
    if any(not math.isfinite(float(row["val_psnr"])) for row in log):
        raise ValueError("Validation PSNR must be finite at every epoch")
    # Python max preserves the first occurrence, matching training's strict > rule.
    selected = max(log, key=lambda row: float(row["val_psnr"]))
    best_epoch, best_psnr = int(selected["epoch"]), float(selected["val_psnr"])
    if summary.get("best_epoch") != best_epoch:
        raise ValueError("Summary best_epoch differs from the frozen validation selection")
    if checkpoints["best"].get("epoch") != best_epoch:
        raise ValueError("best.pt is not the earliest maximum validation-PSNR checkpoint")
    if checkpoints["last"].get("best_epoch") != best_epoch:
        raise ValueError("Last checkpoint saved a different best validation epoch")
    _same_score(summary["best_validation_psnr"], best_psnr, "summary validation PSNR")
    for name, checkpoint in checkpoints.items():
        _same_score(checkpoint["best_psnr"], best_psnr, f"{name} checkpoint validation PSNR")
    per_image = _read_json(run_dir / "best_validation_per_image.json")
    if len(per_image) != SPLIT_COUNTS["validation"]:
        raise ValueError("Best-validation results must contain all 50 pairs")
    if checkpoints["last"].get("best_per_image") != per_image:
        raise ValueError("Last checkpoint best-validation results differ from the saved report")
    _same_score(sum(float(row["psnr"]) for row in per_image) / len(per_image),
                best_psnr, "per-image validation PSNR")
    best_weight = checkpoints["best"]["weight"]
    embedded_weight = checkpoints["last"].get("best_weight")
    if (not isinstance(embedded_weight, dict) or set(embedded_weight) != set(best_weight)
            or any(not torch.equal(embedded_weight[key], value) for key, value in best_weight.items())):
        raise ValueError("best.pt weights differ from the best weights saved in the last checkpoint")
    model = ultrafast_linet_max().cpu().eval()
    model.load_state_dict(best_weight, strict=True)
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Best checkpoint contains non-finite parameters")
    return model, run, {
        "selected_epoch": best_epoch,
        "best_validation_psnr": best_psnr,
        "completed_epochs": 360,
        "checkpoint_sha256": checkpoint_hashes,
        "run_metadata_sha256": sha256(run_dir / "run.json"),
        "training_summary_sha256": sha256(run_dir / "summary.json"),
        "training_log_sha256": sha256(run_dir / "log.csv"),
        "validation_only_selection_verified": True,
    }


def validate_manifests(manifests, run, run_dir):
    """Validate frozen metadata and cross-split isolation without opening images."""
    manifests = Path(manifests)
    summary = _read_json(manifests / "summary.json")
    rows_by_split, hashes = {}, {}
    for split, expected_count in SPLIT_COUNTS.items():
        path = manifests / f"{split}.json"
        hashes[split] = sha256(path)
        if hashes[split] != summary["manifests_sha256"][split]:
            raise ValueError(f"Frozen {split} manifest SHA256 mismatch")
        if split != "test" and hashes[split] != run["manifests_sha256"][split]:
            raise ValueError(f"Training run used a different {split} manifest")
        meta = _read_json(path)
        rows = meta["pairs"]
        if meta.get("dataset") != "LOL-v1" or meta.get("split") != split or len(rows) != expected_count:
            raise ValueError(f"Expected LOL-v1 {split} manifest with {expected_count} pairs")
        for key in ("name", "low", "high"):
            if len({row[key] for row in rows}) != len(rows):
                raise ValueError(f"Duplicate {key} in {split} manifest")
        for row in rows:
            for key in ("low_sha256", "high_sha256"):
                value = row[key]
                if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError(f"Invalid {key} in {split} manifest")
        rows_by_split[split] = rows
    if run.get("train_count") != 435 or run.get("validation_count") != 50:
        raise ValueError("Training run split counts do not match frozen manifests")
    for first, second in itertools.combinations(SPLIT_COUNTS, 2):
        for key in ("name", "low", "high", "low_sha256", "high_sha256"):
            if {row[key] for row in rows_by_split[first]} & {row[key] for row in rows_by_split[second]}:
                raise ValueError(f"Split overlap between {first} and {second}: {key}")
        # Also reject a low image in one split reused as a high image in another.
        first_hashes = {row[key] for row in rows_by_split[first] for key in ("low_sha256", "high_sha256")}
        second_hashes = {row[key] for row in rows_by_split[second] for key in ("low_sha256", "high_sha256")}
        if first_hashes & second_hashes:
            raise ValueError(f"Cross-role image overlap between {first} and {second}")
    recorded_val = _read_json(Path(run_dir) / "best_validation_per_image.json")
    if sorted(row["name"] for row in recorded_val) != sorted(row["name"] for row in rows_by_split["validation"]):
        raise ValueError("Recorded best-validation pairs differ from frozen validation manifest")
    return rows_by_split["test"], hashes


def verify_test_files(rows, data_root):
    """Hash test bytes after the completed-run guard; no image decoding yet."""
    data_root = Path(data_root).resolve()
    for row in rows:
        for key in ("low", "high"):
            relative = Path(row[key])
            path = (data_root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(data_root):
                raise ValueError(f"Dataset path escapes data root: {row[key]}")
            if sha256(path) != row[f"{key}_sha256"]:
                raise ValueError(f"Test image SHA256 mismatch: {row[key]}")


@torch.inference_mode()
def evaluate(data_root, run_dir, output, manifests=ROOT / "manifests", save_images=False):
    """Write one immutable, sigma=0 evaluation. No candidate models or tuning inputs."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new evaluation output directory; existing results are immutable")
    model, run, verification = validate_completed_run(run_dir)
    test_rows, manifest_hashes = validate_manifests(manifests, run, run_dir)
    verify_test_files(test_rows, data_root)
    output.mkdir(parents=True, exist_ok=True)
    if save_images:
        (output / "enhanced").mkdir()
    dataset = ManifestPairs(Path(manifests) / "test.json", data_root)
    scores = []
    for index in range(len(dataset)):
        x, y, name = dataset[index]
        prediction = model(x.unsqueeze(0))[-1].clamp(0, 1)
        target = y.unsqueeze(0)
        if prediction.shape != target.shape or not torch.isfinite(prediction).all():
            raise ValueError(f"Invalid prediction for {name}")
        row = {"name": name, "sigma": 0.0, "width": x.shape[-1], "height": x.shape[-2],
               "psnr": psnr(prediction, target), "ssim": ssim(prediction, target)}
        if not all(math.isfinite(row[key]) for key in ("psnr", "ssim")):
            raise ValueError(f"Non-finite evaluation score for {name}")
        scores.append(row)
        if save_images:
            # Generated outputs only: never copy input/target dataset pixels.
            save_image(prediction, output / "enhanced" / f"{index + 1:02d}_{Path(name).stem}.png")
    aggregate = {key: sum(row[key] for row in scores) / len(scores) for key in ("psnr", "ssim")}
    report = {
        "dataset": "LOL-v1", "split": "official test", "count": len(scores),
        "experiment": "A", "additional_synthetic_noise_sigma": 0.0,
        "noise_condition": "no added synthetic noise; original capture noise may remain",
        "aggregate": aggregate, "per_image": scores, "metrics": METRIC_PROTOCOL,
        "selection": run["selection"], "verification": verification,
        "manifests_sha256": manifest_hashes, "environment": environment(),
        "official_test_used": True, "test_used_for_checkpoint_selection": False,
        "enhanced_images_saved": save_images, "raw_dataset_images_copied": False,
    }
    write_json(output / "metrics.json", report)
    with (output / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    with (output / "aggregate.csv").open("w", newline="") as handle:
        fields = ["experiment", "split", "count", "sigma", "selected_epoch", "psnr", "ssim"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"experiment": "A", "split": "official_test", "count": len(scores),
                         "sigma": 0.0, "selected_epoch": verification["selected_epoch"], **aggregate})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, default=ROOT / "manifests")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--save-images", action="store_true", help="Save generated enhancements only, never raw pairs")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    seed_runtime(0, args.threads)
    report = evaluate(args.data_root, args.run_dir, args.output, args.manifests, args.save_images)
    print(json.dumps({"count": report["count"], "selected_epoch": report["verification"]["selected_epoch"],
                      **report["aggregate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
