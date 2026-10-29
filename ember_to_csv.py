"""
ember_to_csv.py
---------------
Converts EMBER 2018 (v2) raw-feature JSONL files into the dataset.csv format
this malware-detection model expects.

EMBER stores ~2300 features per file; this model uses 15. So we MAP EMBER's
rich JSON fields down to our FEATURE_NAMES. We also drop EMBER's unlabeled
samples (label == -1), keeping only benign (0) and malware (1).

Usage:
    # point at the folder that contains the .jsonl files
    python ember_to_csv.py /path/to/ember2018/

    # or list specific files
    python ember_to_csv.py train_features_0.jsonl test_features.jsonl

    # optional: cap rows for a quick test
    python ember_to_csv.py /path/to/ember2018/ --limit 50000

Output: dataset.csv  (then just run `python train.py`)
"""

from __future__ import annotations
import glob
import json
import math
import os
import sys
import csv

from feature_extractor import FEATURE_NAMES, SUSPICIOUS_APIS

# Match suspicious APIs case-insensitively (EMBER casing can vary)
_SUSPICIOUS_LC = {a.lower() for a in SUSPICIOUS_APIS}


def _entropy_from_histogram(hist) -> float:
    """Shannon entropy (0-8) computed from EMBER's 256-bin byte histogram."""
    if not hist:
        return 0.0
    total = sum(hist)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in hist:
        if c:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def ember_record_to_row(rec: dict):
    """Map one EMBER JSON record -> ordered feature vector + label.
    Returns None if the sample is unlabeled (label == -1)."""
    label = rec.get("label", -1)
    if label not in (0, 1):
        return None  # skip unlabeled / malformed

    general = rec.get("general", {}) or {}
    header = rec.get("header", {}) or {}
    optional = header.get("optional", {}) or {}
    section_info = rec.get("section", {}) or {}
    sections = section_info.get("sections", []) or []
    imports = rec.get("imports", {}) or {}

    # section entropies
    ents = [s.get("entropy", 0.0) for s in sections]
    mean_ent = sum(ents) / len(ents) if ents else 0.0
    max_ent = max(ents) if ents else 0.0

    # imports: dll count, func count, suspicious-API count
    num_dlls = len(imports)
    num_funcs = 0
    num_suspicious = 0
    for funcs in imports.values():
        for fn in funcs:
            num_funcs += 1
            if fn and fn.lower() in _SUSPICIOUS_LC:
                num_suspicious += 1
    if num_funcs == 0:  # fall back to EMBER's scalar count
        num_funcs = int(general.get("imports", 0) or 0)

    # dll_characteristics is a list of flag strings in EMBER
    dll_chars = optional.get("dll_characteristics", []) or []
    aslr = 1 if "DYNAMIC_BASE" in dll_chars else 0
    dep = 1 if "NX_COMPAT" in dll_chars else 0

    # entry point in last section?
    entry_section = section_info.get("entry", "")
    last_name = sections[-1].get("name", "") if sections else ""
    entry_in_last = 1 if (entry_section and entry_section == last_name) else 0

    feats = {
        "file_size": int(general.get("size", 0) or 0),
        "byte_entropy": _entropy_from_histogram(rec.get("histogram", [])),
        "is_pe": 1,  # every EMBER sample is a PE file
        "num_sections": len(sections),
        "mean_section_entropy": mean_ent,
        "max_section_entropy": max_ent,
        "num_imported_dlls": num_dlls,
        "num_imported_funcs": num_funcs,
        "num_suspicious_apis": num_suspicious,
        "num_exports": int(general.get("exports", 0) or 0),
        "has_signature": int(general.get("has_signature", 0) or 0),
        "aslr_enabled": aslr,
        "dep_enabled": dep,
        "size_of_code": int(optional.get("sizeof_code", 0) or 0),
        "entry_in_last_section": entry_in_last,
    }
    return [feats[name] for name in FEATURE_NAMES] + [label]


def resolve_inputs(args):
    """Accept a directory (glob *.jsonl) or explicit file paths."""
    files = []
    for a in args:
        if os.path.isdir(a):
            files.extend(sorted(glob.glob(os.path.join(a, "*.jsonl"))))
        else:
            files.append(a)
    return files


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = None
    for a in sys.argv[1:]:
        if a.startswith("--limit"):
            # supports "--limit 1000" or "--limit=1000"
            if "=" in a:
                limit = int(a.split("=")[1])
            else:
                idx = sys.argv.index(a)
                limit = int(sys.argv[idx + 1])

    if not args:
        print(__doc__)
        sys.exit(1)

    files = resolve_inputs(args)
    if not files:
        print("No .jsonl files found in the given path(s).")
        sys.exit(1)

    print(f"Reading {len(files)} file(s):")
    for f in files:
        print(f"  - {f}")

    kept = skipped = 0
    with open("dataset.csv", "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(FEATURE_NAMES + ["label"])
        for path in files:
            with open(path, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    row = ember_record_to_row(rec)
                    if row is None:
                        skipped += 1
                        continue
                    writer.writerow(row)
                    kept += 1
                    if kept % 50000 == 0:
                        print(f"  ...{kept} rows written")
                    if limit and kept >= limit:
                        break
            if limit and kept >= limit:
                break

    print(f"\nDone. Wrote dataset.csv with {kept} labeled rows "
          f"({skipped} skipped/unlabeled).")
    print("Next: python train.py")


if __name__ == "__main__":
    main()
