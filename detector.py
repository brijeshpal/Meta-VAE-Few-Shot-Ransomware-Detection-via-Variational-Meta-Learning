"""
detector.py
-----------
Loads the trained model and gives a verdict on a file the moment it appears.

Usage:
    python detector.py path/to/file1 path/to/file2 ...

Outputs a probability, a verdict, and the top reasons that drove the score.
"""

from __future__ import annotations
import sys
import joblib
import numpy as np

from feature_extractor import extract_features, features_to_vector, FEATURE_NAMES

# Probability above which we call something malware. Tune per environment:
# raise it to cut false positives, lower it to catch more (more false alarms).
THRESHOLD = 0.5


class Detector:
    def __init__(self, model_path: str = "model.joblib"):
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.features = bundle["features"]
        # Last pipeline step is the classifier, whatever it's named
        # (RandomForest from train.py, XGBoost from train_gpu.py).
        clf = self.pipeline.steps[-1][1]
        self._importances = dict(zip(self.features, clf.feature_importances_))

    def scan(self, path: str) -> dict:
        feats = extract_features(path)
        vec = np.array(features_to_vector(feats)).reshape(1, -1)
        prob = float(self.pipeline.predict_proba(vec)[0, 1])
        verdict = "MALWARE" if prob >= THRESHOLD else "benign"

        # Explain: which high-importance features look malware-like here
        reasons = self._reasons(feats)
        return {"path": path, "probability": prob, "verdict": verdict,
                "reasons": reasons, "features": feats}

    def _reasons(self, feats: dict) -> list[str]:
        notes = []
        if feats.get("max_section_entropy", 0) > 7.2:
            notes.append("very high section entropy (likely packed/encrypted)")
        if feats.get("num_suspicious_apis", 0) >= 6:
            notes.append(f"imports {feats['num_suspicious_apis']} commonly-abused APIs")
        if feats.get("is_pe") and not feats.get("has_signature"):
            notes.append("executable is not digitally signed")
        if feats.get("entry_in_last_section"):
            notes.append("entry point in last section (packer signature)")
        if feats.get("is_pe") and feats.get("num_imported_funcs", 0) < 15:
            notes.append("unusually few imports (possible obfuscation)")
        if not notes:
            notes.append("no strong individual red flags; verdict from combined profile")
        return notes


def _print(result: dict) -> None:
    p = result["probability"]
    bar = "#" * int(p * 30)
    flag = "[!]" if result["verdict"] == "MALWARE" else "[ ]"
    print(f"\n{flag} {result['path']}")
    print(f"    verdict     : {result['verdict']}  (p={p:.3f})")
    print(f"    risk        : [{bar:<30}]")
    print(f"    reasons     :")
    for r in result["reasons"]:
        print(f"        - {r}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python detector.py <file> [<file> ...]")
        sys.exit(1)
    det = Detector()
    for path in sys.argv[1:]:
        try:
            _print(det.scan(path))
        except Exception as e:
            print(f"\n[?] {path}\n    error: {e}")
