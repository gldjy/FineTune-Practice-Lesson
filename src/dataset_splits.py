"""Locates the train/val/test JSONL files inside a dataset directory."""

import glob
import os
import re

# 文件名按非字母数字切成词之后，命中这些词就归到对应子集
SPLIT_KEYWORDS = {
    "train": {"train"},
    "val": {"val", "valid", "validation", "dev"},
    "test": {"test"},
}


def dataset_split_of(path):
    """Guesses which split a JSONL file name belongs to, or None when unsure."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    tokens = set(re.split(r"[^a-z0-9]+", stem))
    for split, keywords in SPLIT_KEYWORDS.items():
        if tokens & keywords:
            return split
    return None


def find_split_file(dataset_dir, split):
    """Finds the single JSONL file of one split in dataset_dir."""
    paths = sorted(glob.glob(os.path.join(dataset_dir, "*.jsonl")))
    matches = [path for path in paths if dataset_split_of(path) == split]
    if len(matches) == 1:
        return matches[0]
    found = ", ".join(os.path.basename(path) for path in paths) or "no .jsonl files"
    if not matches:
        raise SystemExit(f"No {split} file found in {dataset_dir}; found: {found}")
    names = ", ".join(os.path.basename(path) for path in matches)
    raise SystemExit(
        f"More than one {split} file in {dataset_dir}: {names}; keep only one of them"
    )
