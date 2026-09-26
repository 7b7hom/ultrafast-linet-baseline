"""Scientific-contract checks: parity, gradients, disjoint data and paired crops."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import (ManifestPairs, UltraFastLiNETLoss, load_official, psnr, ssim,
                    ultrafast_linet_max, validate_manifest_pair)
from prepare_data import grouped_partition, index_images, pair_rows
from dsconv import shift_aggregate

torch.set_num_threads(1)


def test_pinned_sources_are_unmodified():
    provenance = json.loads((ROOT / "provenance.json").read_text())
    for filename, expected in provenance["files"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected


def test_official_checkpoint_and_odd_shape():
    model, _ = load_official(ROOT / "weights/official_max.pkl")
    assert sum(p.numel() for p in model.parameters()) == 180
    with torch.inference_mode():
        outputs = model(torch.rand(1, 3, 181, 183) * .05)
    assert [list(o.shape[-2:]) for o in outputs] == [[46, 46], [91, 92], [181, 183]]
    assert all(torch.isfinite(o).all() for o in outputs)


def test_shift_equals_fixed_dilated_depthwise_convolution():
    torch.manual_seed(2)
    x = torch.randn(1, 3, 31, 33)
    for distance in (1, 3, 5):
        reference = torch.nn.functional.conv2d(x, torch.ones(3, 1, 3, 3),
                                              padding=distance, dilation=distance, groups=3)
        torch.testing.assert_close(shift_aggregate(x, distance), reference)


def test_loss_backpropagates_to_all_180_parameters():
    torch.manual_seed(42)
    model = ultrafast_linet_max()
    x, y = torch.rand(1, 3, 180, 180) * .1, torch.rand(1, 3, 180, 180)
    loss, parts = UltraFastLiNETLoss()(model(x), y)
    assert torch.isfinite(loss)
    assert set(parts) == {"rec", "ms_ssim", "grad"}
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    before = [p.detach().clone() for p in model.parameters()]
    torch.optim.Adam(model.parameters(), lr=.01).step()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_same_seed_same_initialization():
    torch.manual_seed(42)
    a = ultrafast_linet_max()
    torch.manual_seed(42)
    b = ultrafast_linet_max()
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


def test_metrics_identity_and_known_error():
    x = torch.full((1, 3, 180, 180), .2)
    assert psnr(x, x + .1) == pytest.approx(20., abs=1e-5)
    assert ssim(x, x) == pytest.approx(1., abs=1e-6)


def test_paired_center_crop_and_test_rejection(tmp_path):
    arr = np.random.default_rng(4).integers(0, 256, (200, 204, 3), dtype=np.uint8)
    Image.fromarray(arr).save(tmp_path / "low.png")
    Image.fromarray(arr).save(tmp_path / "high.png")
    row = {"name": "pair", "low": "low.png", "high": "high.png", "low_sha256": "a", "high_sha256": "b"}
    path = tmp_path / "train.json"
    path.write_text(json.dumps({"split": "train", "pairs": [row]}))
    dataset = ManifestPairs(path, tmp_path, 180)
    x, y, _ = dataset[0]
    assert x.shape == (3, 180, 180) and torch.equal(x, y)
    reference = torch.from_numpy(arr[10:190, 12:192].copy()).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(x, reference)
    test_path = tmp_path / "test.json"
    test_path.write_text(json.dumps({"split": "test", "pairs": [row]}))
    with pytest.raises(ValueError, match="never official test"):
        validate_manifest_pair(dataset, ManifestPairs(test_path, tmp_path))


def test_unpaired_and_duplicate_names_fail(tmp_path):
    for kind in ("low", "high"):
        (tmp_path / "train" / kind).mkdir(parents=True)
    img = Image.new("RGB", (180, 180))
    img.save(tmp_path / "train/low/a.png")
    img.save(tmp_path / "train/high/b.png")
    with pytest.raises(ValueError, match="Unpaired"):
        pair_rows(tmp_path, "train", 1, True)
    (tmp_path / "train/low/nested").mkdir()
    img.save(tmp_path / "train/low/nested/a.png")
    with pytest.raises(ValueError, match="Duplicate"):
        index_images(tmp_path / "train/low")


def test_duplicate_targets_cannot_cross_splits():
    rows = [{"name": str(i), "low_sha256": f"low{i}", "high_sha256": f"high{i // 2}"}
            for i in range(100)]
    first, groups = grouped_partition(rows, 42, 50)
    second, _ = grouped_partition(rows, 42, 50)
    assert first == second and len(first) == 50 and len(groups) == 50
    assert not ({rows[i]["high_sha256"] for i in first} &
                {rows[i]["high_sha256"] for i in set(range(100)) - first})


def test_operation_counter_includes_shift_arithmetic():
    from reproduce import operation_counts
    model = ultrafast_linet_max().eval()
    result = operation_counts(model, torch.zeros(1, 3, 180, 180))
    assert result["shift_adds"] > 0 and result["skip_adds"] > 0
    assert result["nominal_arithmetic_flops"] > 2 * result["conv_macs"]
    assert result["thop_partial_ops_not_total_flops"] > 0
    assert sum(p.numel() for p in model.parameters()) == 180
