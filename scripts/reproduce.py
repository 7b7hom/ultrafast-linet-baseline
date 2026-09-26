"""CPU reproduction on ONE training image, with timing and explicit FLOP conventions."""
import argparse
import copy
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
# Activate setuptools' Python 3.12 distutils shim before the legacy THOP import.
import setuptools  # noqa: F401
from thop import profile

from common import (ROOT, environment, load_official, psnr, save_image,
                    seed_runtime, sha256, ssim, tensor_image, write_json)
from dsconv import DSConv, MSRB


def operation_counts(model, x):
    """Shape-driven nominal arithmetic counts for the full three-output forward.

    One multiply or add = one FLOP; bias additions counted separately. Pool
    padding is counted as a nominal 3x3 window. Sigmoid, ReLU, memory moves,
    indexing, nearest interpolation and allocation are listed/excluded.
    """
    counts = {k: 0 for k in ("conv_macs", "conv_bias_adds", "pool_nominal_flops",
                            "shift_adds", "gate_multiplies", "branch_adds",
                            "skip_adds", "relu_elements", "sigmoid_elements")}

    def hook(module, inputs, output):
        n = output.numel()
        if isinstance(module, torch.nn.Conv2d):
            counts["conv_macs"] += n * (module.in_channels // module.groups) * np.prod(module.kernel_size).item()
            counts["conv_bias_adds"] += n if module.bias is not None else 0
        elif isinstance(module, torch.nn.AvgPool2d):
            # 8 additions + 1 division per nominal 3x3 window.
            counts["pool_nominal_flops"] += 9 * n
        elif isinstance(module, torch.nn.ReLU):
            counts["relu_elements"] += n
        elif isinstance(module, DSConv):
            h, w = inputs[0].shape[-2:]
            if module.mode == "down":
                h, w = (h + 1) // 2, (w + 1) // 2
            elements = inputs[0].shape[0] * 3 * h * w
            counts["shift_adds"] += 8 * elements
            counts["gate_multiplies"] += elements
            counts["sigmoid_elements"] += elements
            if module.mode == "up":
                counts["skip_adds"] += n
        elif isinstance(module, MSRB):
            counts["branch_adds"] += (len(module._names) - 1) * n

    supported = (torch.nn.Conv2d, torch.nn.AvgPool2d, torch.nn.ReLU, DSConv, MSRB)
    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, supported)]
    try:
        with torch.inference_mode():
            outputs = model(x)
    finally:
        for handle in handles:
            handle.remove()
    counts["skip_adds"] += sum(t.numel() for t in outputs)
    keys = ("conv_bias_adds", "pool_nominal_flops", "shift_adds", "gate_multiplies", "branch_adds", "skip_adds")
    counts["nominal_arithmetic_flops"] = 2 * counts["conv_macs"] + sum(counts[k] for k in keys)
    counts["convention"] = "Full 3-output forward, multiply/add=1 each; pool nominal 3x3 including padding; excludes sigmoid/ReLU, indexing, copies, nearest interpolation, allocation"
    with torch.inference_mode():
        partial, _ = profile(copy.deepcopy(model), inputs=(x,), verbose=False)
    counts["thop_partial_ops_not_total_flops"] = partial
    counts["thop_limitation"] = "Default THOP omits functional shift additions, gating, skip/branch additions; do not report as total FLOPs"
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--target", type=Path)
    p.add_argument("--weights", type=Path, default=ROOT / "weights/official_max.pkl")
    p.add_argument("--output", type=Path, default=ROOT / "reports/reproduction")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args()
    if args.warmup < 1 or args.repeats < 2 or args.threads < 1:
        p.error("warmup/threads >= 1 and repeats >= 2 are required")
    seed_runtime(42, args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_official(args.weights)
    model.eval()
    x = tensor_image(args.input).unsqueeze(0)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x)
        elapsed = []
        for _ in range(args.repeats):
            t0 = time.perf_counter_ns()
            outputs = model(x)
            elapsed.append((time.perf_counter_ns() - t0) / 1e6)
        raw = outputs[-1]
        assert raw.shape == x.shape and torch.isfinite(raw).all()
        prediction = raw.clamp(0, 1)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (1024 ** 2 if sys.platform == "darwin" else 1024)
    report = {"purpose": "released-checkpoint diagnostic on a training image; not independent validation or A baseline",
              "environment": environment(), "device": "cpu", "dtype": str(x.dtype),
              "input_path": str(args.input.resolve()), "input_sha256": sha256(args.input),
              "checkpoint_sha256": sha256(args.weights), "checkpoint_epoch_metadata": checkpoint.get("epoch"),
              "parameters": sum(v.numel() for v in model.parameters()),
              "trainable_parameters": sum(v.numel() for v in model.parameters() if v.requires_grad),
              "parameter_tensor_bytes": sum(v.numel() * v.element_size() for v in model.parameters()),
              "checkpoint_file_bytes": args.weights.stat().st_size,
              "input_shape_nchw": list(x.shape), "decoder_shapes": [list(v.shape) for v in outputs],
              "input_range": [x.min().item(), x.max().item()],
              "raw_output_range": [raw.min().item(), raw.max().item()],
              "clipped_pixel_fraction": ((raw < 0) | (raw > 1)).float().mean().item(),
              "saved_output_range": [prediction.min().item(), prediction.max().item()],
              "latency": {"scope": "CPU model forward only, all 3 decoder outputs; excludes decode, tensor conversion, clamp, metrics, save",
                          "warmup": args.warmup, "repeats": args.repeats, "batch_size": 1,
                          "mean_ms": statistics.mean(elapsed), "median_ms": statistics.median(elapsed),
                          "p95_ms": float(np.percentile(elapsed, 95)), "samples_ms": elapsed},
              "memory": {"process_peak_rss_mib_after_timing": peak_mib,
                         "scope": "Process high-water RSS including Python, PyTorch, loaded libraries, weights, activations; not isolated model memory"}}
    save_image(x, args.output / "input.png")
    save_image(prediction, args.output / "enhanced.png")
    panels = [("Input: no added synthetic noise", "input.png"), ("Official checkpoint (diagnostic)", "enhanced.png")]
    if args.target:
        target = tensor_image(args.target).unsqueeze(0)
        if target.shape != x.shape:
            raise ValueError("Input/target shape mismatch")
        report["single_training_image_metrics_not_benchmark"] = {
            "input_psnr": psnr(x, target), "input_ssim": ssim(x, target),
            "enhanced_psnr": psnr(prediction, target), "enhanced_ssim": ssim(prediction, target),
            "protocol": "RGB float32 [0,1], output clamped, per-image full-resolution, no border crop, no GT brightness correction; pytorch_msssim.ssim default 11x11 sigma1.5 window"}
        save_image(target, args.output / "target.png")
        panels.append(("Normal-light reference", "target.png"))
    w, h = x.shape[-1], x.shape[-2]
    canvas = Image.new("RGB", (w * len(panels), h + 38), "#18222d")
    draw = ImageDraw.Draw(canvas)
    for i, (label, filename) in enumerate(panels):
        with Image.open(args.output / filename) as panel:
            canvas.paste(panel, (i * w, 38))
        draw.text((i * w + 12, 12), label, fill="white")
    canvas.save(args.output / "comparison.png")
    report["operations"] = operation_counts(model, x)
    write_json(args.output / "metrics.json", report)
    print(json.dumps({k: report[k] for k in ("parameters", "input_shape_nchw", "raw_output_range", "memory")}, indent=2))
    print(f"CPU median: {report['latency']['median_ms']:.3f} ms; p95: {report['latency']['p95_ms']:.3f} ms")
    print(f"Saved {args.output / 'metrics.json'}")


if __name__ == "__main__":
    main()
