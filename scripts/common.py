"""Shared runtime. The pinned upstream model and loss are imported unchanged."""
import hashlib
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from model import ultrafast_linet_max  # noqa: E402
from losses import UltraFastLiNETLoss  # noqa: E402
from metrics import psnr, ssim  # noqa: E402


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_runtime(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def environment():
    cpu = platform.processor()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                          text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            cpu = platform.machine() + " (CPU brand unavailable in sandbox)"
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "platform": platform.platform(), "cpu": cpu,
            "torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads()}


def tensor_image(path):
    with Image.open(path) as image:
        return TF.to_tensor(image.convert("RGB"))


def save_image(x, path):
    x = x.detach().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    # Round consistently with torchvision.utils.save_image.
    arr = (x.permute(1, 2, 0) * 255 + 0.5).clamp(0, 255).to(torch.uint8).numpy()
    Image.fromarray(arr).save(path)


def load_official(path):
    model = ultrafast_linet_max().cpu()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint.get("weight", checkpoint), strict=True)
    return model, checkpoint


class ManifestPairs(Dataset):
    def __init__(self, manifest, data_root, crop=None):
        self.path = Path(manifest)
        self.meta = json.loads(self.path.read_text())
        self.rows = self.meta["pairs"]
        self.data_root = Path(data_root)
        self.crop = crop
        if not self.rows:
            raise ValueError("Empty manifest")
        if crop is not None and crop <= 160:
            raise ValueError("Default five-scale MS-SSIM requires crop > 160; use 180")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x, y = (tensor_image(self.data_root / row[k]) for k in ("low", "high"))
        if x.shape != y.shape:
            raise ValueError(f"Pair shape mismatch: {row['name']}")
        if self.crop:
            if min(x.shape[-2:]) < self.crop:
                raise ValueError(f"Image smaller than crop: {row['name']}")
            x, y = TF.center_crop(x, self.crop), TF.center_crop(y, self.crop)
        return x, y, row["name"]


def validate_manifest_pair(train, val):
    if train.meta["split"] != "train" or val.meta["split"] != "validation":
        raise ValueError("Training requires train and validation manifests, never official test")
    for key in ("name", "low_sha256", "high_sha256"):
        if {r[key] for r in train.rows} & {r[key] for r in val.rows}:
            raise ValueError(f"Train/validation overlap: {key}")
