"""
synthetic_data.py
-----------------
Generates a LABELED feature dataset for demonstration.

IMPORTANT: This is synthetic. It samples feature distributions that reflect
well-documented statistical differences between malware and benign software
(e.g. malware is more often packed -> higher section entropy, more often
unsigned, imports more injection/persistence APIs). It lets the whole pipeline
train and run end-to-end WITHOUT handling real malware.

For real-world deployment, replace this with features extracted from a real
labeled corpus such as EMBER (github.com/elastic/ember) or SOREL-20M.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from feature_extractor import FEATURE_NAMES


def _sample_benign(rng: np.random.Generator) -> dict:
    return {
        "file_size": int(abs(rng.normal(1_800_000, 1_200_000)) + 20_000),
        "byte_entropy": float(np.clip(rng.normal(5.6, 0.8), 0, 8)),
        "is_pe": 1,
        "num_sections": int(np.clip(rng.normal(5, 1.2), 2, 12)),
        "mean_section_entropy": float(np.clip(rng.normal(4.8, 0.7), 0, 8)),
        "max_section_entropy": float(np.clip(rng.normal(6.0, 0.6), 0, 8)),
        "num_imported_dlls": int(np.clip(rng.normal(9, 3), 1, 40)),
        "num_imported_funcs": int(np.clip(rng.normal(140, 60), 5, 800)),
        "num_suspicious_apis": int(np.clip(rng.normal(2, 1.5), 0, 12)),
        "num_exports": int(np.clip(rng.normal(8, 20), 0, 400)),
        "has_signature": int(rng.random() < 0.82),     # most clean SW is signed
        "aslr_enabled": int(rng.random() < 0.93),
        "dep_enabled": int(rng.random() < 0.95),
        "size_of_code": int(abs(rng.normal(600_000, 400_000)) + 4_000),
        "entry_in_last_section": int(rng.random() < 0.04),
    }


def _sample_malware(rng: np.random.Generator) -> dict:
    return {
        "file_size": int(abs(rng.normal(550_000, 500_000)) + 8_000),
        "byte_entropy": float(np.clip(rng.normal(7.0, 0.7), 0, 8)),   # packed
        "is_pe": 1,
        "num_sections": int(np.clip(rng.normal(4, 1.6), 1, 12)),
        "mean_section_entropy": float(np.clip(rng.normal(6.7, 0.8), 0, 8)),
        "max_section_entropy": float(np.clip(rng.normal(7.6, 0.4), 0, 8)),
        "num_imported_dlls": int(np.clip(rng.normal(4, 2.5), 1, 40)),
        "num_imported_funcs": int(np.clip(rng.normal(45, 40), 1, 800)),
        "num_suspicious_apis": int(np.clip(rng.normal(7, 3), 0, 20)),  # higher
        "num_exports": int(np.clip(rng.normal(2, 6), 0, 400)),
        "has_signature": int(rng.random() < 0.12),     # usually unsigned
        "aslr_enabled": int(rng.random() < 0.45),
        "dep_enabled": int(rng.random() < 0.50),
        "size_of_code": int(abs(rng.normal(180_000, 200_000)) + 1_000),
        "entry_in_last_section": int(rng.random() < 0.38),  # packer tell
    }


def generate(n_per_class: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_per_class):
        b = _sample_benign(rng); b["label"] = 0; rows.append(b)
        m = _sample_malware(rng); m["label"] = 1; rows.append(m)
    df = pd.DataFrame(rows)
    return df[FEATURE_NAMES + ["label"]].sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    df.to_csv("dataset.csv", index=False)
    print(f"Wrote dataset.csv  shape={df.shape}")
    print(df["label"].value_counts().rename({0: "benign", 1: "malware"}))
