"""Fresh A training using fixed train/validation manifests; official test is inaccessible here."""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import (ROOT, ManifestPairs, UltraFastLiNETLoss, environment, psnr,
                    seed_runtime, sha256, ssim, ultrafast_linet_max,
                    validate_manifest_pair, write_json)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def best_checkpoint(last):
    return {"weight": last["best_weight"], "epoch": last["best_epoch"],
            "best_psnr": last["best_psnr"], "config": last["config"],
            "pilot": last["pilot"]}


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    rows = []
    for x, y, names in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)[-1].clamp(0, 1)
        rows.append({"name": names[0], "psnr": psnr(out, y), "ssim": ssim(out, y)})
    return {key: sum(r[key] for r in rows) / len(rows) for key in ("psnr", "ssim")}, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifests", type=Path, default=ROOT / "manifests")
    p.add_argument("--config", type=Path, default=ROOT / "configs/baseline_a.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--device", choices=("cpu", "cuda", "mps"))
    p.add_argument("--pilot", action="store_true", help="Label this short run as a smoke/pilot, never a completed baseline")
    p.add_argument("--resume", type=Path, help="Resume this run's last.pt at an epoch boundary")
    p.add_argument("--stop-after-epoch", type=int, help="Optional orderly pause; does not change the planned schedule")
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.device:
        cfg["device"] = args.device
    if cfg["experiment"] != "A" or cfg["additional_noise_probability"] != 0 or cfg["selection"] != "validation_psnr_sigma0":
        raise ValueError("This entrypoint implements clean-input baseline A only")
    if cfg["epochs"] <= 0 or cfg["batch_size"] <= 0:
        raise ValueError("epochs/batch_size must be positive")
    if args.stop_after_epoch is not None and args.stop_after_epoch <= 0:
        raise ValueError("stop-after-epoch must be positive")
    if args.resume and args.resume.resolve() != (args.output / "last.pt").resolve():
        raise ValueError("Resume must use last.pt from the same output directory")
    if not args.resume and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory; existing runs are immutable")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_runtime(cfg["seed"], cfg["threads"])
    train = ManifestPairs(args.manifests / "train.json", args.data_root, cfg["crop"])
    val = ManifestPairs(args.manifests / "validation.json", args.data_root)
    validate_manifest_pair(train, val)
    for ds in (train, val):
        for row in ds.rows:
            for key in ("low", "high"):
                if sha256(args.data_root / row[key]) != row[f"{key}_sha256"]:
                    raise ValueError(f"Dataset changed since manifest creation: {row[key]}")
    device = torch.device(cfg["device"])
    model = ultrafast_linet_max().to(device)
    assert sum(v.numel() for v in model.parameters()) == 180
    initial_path = args.output / "initial.pt"
    if not args.resume:
        atomic_save({"weight": model.state_dict(), "seed": cfg["seed"], "kind": "fresh_random_initialization"}, initial_path)
    loader_generator = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, drop_last=False,
                        num_workers=cfg["num_workers"], generator=loader_generator)
    validation_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=0)
    criterion = UltraFastLiNETLoss(grad_weights=cfg["grad_weights"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_step"], gamma=cfg["lr_gamma"])
    metadata = {"config": cfg, "pilot": args.pilot, "environment": environment(),
                "parameters": 180, "train_count": len(train), "validation_count": len(val),
                "initialization": "fresh; official checkpoint never loaded", "initial_file_sha256": sha256(initial_path),
                "manifests_sha256": {s: sha256(args.manifests / f"{s}.json") for s in ("train", "validation")},
                "provenance": json.loads((ROOT / "provenance.json").read_text()),
                "project_source_sha256": {name: sha256(ROOT / "scripts" / name) for name in ("train_a.py", "common.py")},
                "selection": "max mean per-image validation PSNR, no added synthetic noise; ties keep earlier epoch",
                "official_test_used": False,
                "metrics": "RGB float32 [0,1], clamp prediction only; full resolution; no border crop or brightness matching"}
    best = -math.inf
    best_epoch, start_epoch, best_weight = 0, 0, None
    best_per_image = None
    fields = ["epoch", "loss", "rec", "ms_ssim", "grad", "val_psnr", "val_ssim", "lr", "train_seconds", "val_seconds", "images_seen"]
    if args.resume:
        previous = json.loads((args.output / "run.json").read_text())
        for key in ("config", "pilot", "manifests_sha256", "provenance", "initial_file_sha256", "project_source_sha256"):
            if previous[key] != metadata[key]:
                raise ValueError(f"Cannot resume changed experiment: {key}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["config"] != cfg or checkpoint["pilot"] != args.pilot:
            raise ValueError("Checkpoint configuration differs from this run")
        model.load_state_dict(checkpoint["weight"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        loader_generator.set_state(checkpoint["loader_generator_state"])
        if device.type == "cuda" and "cuda_rng_states" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        start_epoch, best = checkpoint["epoch"], checkpoint["best_psnr"]
        best_epoch, best_weight = checkpoint["best_epoch"], checkpoint["best_weight"]
        best_per_image = checkpoint["best_per_image"]
        with open(args.output / "log.csv", newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= start_epoch]
        if [int(r["epoch"]) for r in rows] != list(range(1, start_epoch + 1)):
            raise ValueError("Checkpoint/log epoch history is incomplete")
        with open(args.output / "log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        atomic_save(best_checkpoint(checkpoint), args.output / "best.pt")
        write_json(args.output / "best_validation_per_image.json", best_per_image)
        print(f"Resuming at epoch {start_epoch + 1}", flush=True)
    else:
        write_json(args.output / "run.json", metadata)
    final_epoch = min(cfg["epochs"], args.stop_after_epoch or cfg["epochs"])
    if final_epoch < start_epoch:
        raise ValueError("Stop epoch precedes resumed checkpoint")
    with open(args.output / "log.csv", "a" if args.resume else "w", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=fields)
        if not args.resume:
            writer.writeheader()
        for epoch in range(start_epoch + 1, final_epoch + 1):
            model.train()
            sums = dict.fromkeys(("loss", "rec", "ms_ssim", "grad"), 0.0)
            seen = 0
            start = time.perf_counter()
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(x)
                loss, parts = criterion(outputs, y)  # raw output, no training clamp
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"Missing/non-finite gradient: {name}")
                optimizer.step()
                n = x.shape[0]
                seen += n
                sums["loss"] += loss.item() * n
                for key, value in parts.items():
                    sums[key] += value * n
            train_seconds = time.perf_counter() - start
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            start = time.perf_counter()
            scores, per_image = validate(model, validation_loader, device)
            val_seconds = time.perf_counter() - start
            improved = scores["psnr"] > best
            if improved:
                best = scores["psnr"]
                best_epoch = epoch
                best_weight = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_per_image = per_image
            checkpoint = {"weight": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "epoch": epoch, "best_psnr": best,
                          "config": cfg, "pilot": args.pilot, "torch_rng_state": torch.get_rng_state(),
                          "loader_generator_state": loader_generator.get_state(),
                          "best_epoch": best_epoch, "best_weight": best_weight, "best_per_image": best_per_image}
            if device.type == "cuda":
                checkpoint["cuda_rng_states"] = torch.cuda.get_rng_state_all()
            row = {"epoch": epoch, **{k: v / seen for k, v in sums.items()},
                   "val_psnr": scores["psnr"], "val_ssim": scores["ssim"], "lr": lr,
                   "train_seconds": train_seconds, "val_seconds": val_seconds, "images_seen": seen}
            writer.writerow(row)
            log.flush()
            os.fsync(log.fileno())
            atomic_save(checkpoint, args.output / "last.pt")
            if improved:
                atomic_save(best_checkpoint(checkpoint), args.output / "best.pt")
                write_json(args.output / "best_validation_per_image.json", per_image)
            print(json.dumps(row), flush=True)
    write_json(args.output / "summary.json", {"completed_epochs": final_epoch, "planned_epochs": cfg["epochs"],
               "best_validation_psnr": best, "best_epoch": best_epoch,
               "pilot": args.pilot, "full_baseline_complete": not args.pilot and final_epoch == cfg["epochs"] == 360,
               "checkpoint_sha256": {k: sha256(args.output / f"{k}.pt") for k in ("best", "last")},
               "official_test_used": False})


if __name__ == "__main__":
    main()
