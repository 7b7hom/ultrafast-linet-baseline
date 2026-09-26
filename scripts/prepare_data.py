"""Check LOL-v1 pairs and freeze 435/50/15 manifests without inspecting test pixels."""
import argparse
import json
import random
from pathlib import Path

from PIL import Image
from common import sha256, write_json

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def grouped_partition(rows, seed, validation_count=50):
    """Keep pairs sharing an identical low OR high file in the same split.

    Connected components also cover transitive duplicates. A deterministic
    subset-sum selects exactly validation_count images without splitting groups.
    This catches byte-identical duplicates, not all semantically similar scenes.
    """
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in ("low_sha256", "high_sha256"):
            digest = row[key]
            if digest in seen:
                parents[find(i)] = find(seen[digest])
            seen[digest] = i
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    choices = {0: ()}
    for gi, group in enumerate(groups):
        for total, selected in list(choices.items()):
            target = total + len(group)
            if target <= validation_count and target not in choices:
                choices[target] = selected + (gi,)
        if validation_count in choices:
            break
    if validation_count not in choices:
        raise ValueError("Cannot reach exact validation size while keeping duplicate groups intact")
    val = {i for gi in choices[validation_count] for i in groups[gi]}
    duplicates = [[rows[i]["name"] for i in group] for group in groups if len(group) > 1]
    return val, duplicates


def index_images(directory):
    result = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.name in result:
                raise ValueError(f"Duplicate filename: {path.name}")
            result[path.name] = path
    if not result:
        raise ValueError(f"No images: {directory}")
    return result


def pair_rows(root, relative, expected, inspect_pixels):
    low, high = (index_images(root / relative / k) for k in ("low", "high"))
    if low.keys() != high.keys():
        raise ValueError(f"Unpaired images: {sorted(low.keys() ^ high.keys())}")
    if len(low) != expected:
        raise ValueError(f"Expected {expected} pairs under {relative}, got {len(low)}")
    rows = []
    for name in sorted(low):
        row = {"name": name}
        for kind, files in (("low", low), ("high", high)):
            path = files[name]
            row[kind] = str(path.relative_to(root))
            row[f"{kind}_sha256"] = sha256(path)
        if inspect_pixels:
            with Image.open(low[name]) as a, Image.open(high[name]) as b:
                if a.size != b.size or a.mode != "RGB" or b.mode != "RGB":
                    raise ValueError(f"Pair dimensions/RGB mismatch: {name}")
                if min(a.size) < 180:
                    raise ValueError(f"Pair smaller than crop: {name}")
                row["width"], row["height"] = a.size
                a.verify()
                b.verify()
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--train-dir", default="our485")
    p.add_argument("--test-dir", default="eval15")
    p.add_argument("--output", type=Path, default=Path("manifests"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train = pair_rows(args.data_root, args.train_dir, 485, True)
    test = pair_rows(args.data_root, args.test_dir, 15, False)
    for key in ("name", "low_sha256", "high_sha256"):
        if {r[key] for r in train} & {r[key] for r in test}:
            raise ValueError(f"Official train/test overlap: {key}")
    val_indices, duplicate_groups = grouped_partition(train, args.seed)
    splits = {"train": [r for i, r in enumerate(train) if i not in val_indices],
              "validation": [r for i, r in enumerate(train) if i in val_indices],
              "test": test}
    for split, rows in splits.items():
        doc = {"dataset": "LOL-v1", "split": split, "split_seed": args.seed,
               "partition_algorithm": "Sorted filenames -> connected components sharing low/high SHA256 -> seeded group shuffle -> deterministic exact-50 subset-sum",
               "test_pixels_decoded": False, "pairs": rows}
        path = args.output / f"{split}.json"
        if path.exists() and json.loads(path.read_text()) != doc:
            raise FileExistsError(f"Refusing to change frozen manifest: {path}")
        write_json(path, doc)
    summary = {"counts": {k: len(v) for k, v in splits.items()}, "seed": args.seed,
               "duplicate_groups_kept_together": duplicate_groups,
               "limitation": "Byte-identical duplicates grouped; near-duplicate scene identity not independently annotated",
               "manifests_sha256": {k: sha256(args.output / f"{k}.json") for k in splits},
               "official_test_access": "filenames and file hashes only; no decoding, inference, metrics, selection"}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
