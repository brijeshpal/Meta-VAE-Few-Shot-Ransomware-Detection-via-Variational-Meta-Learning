"""
train.py
--------
Trains the malware-detection classifier and saves it to model.joblib.

Pipeline = StandardScaler + RandomForest. Random forests handle the mixed
numeric/flag features well, are robust to outliers, and give feature
importances we can inspect to understand WHY the model flags something.
"""

from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_extractor import FEATURE_NAMES
import synthetic_data


def main(model_path: str = "model.joblib"):
    # 1. Load data (generate synthetic if no real dataset.csv is present)
    try:
        df = pd.read_csv("dataset.csv")
        print("Loaded dataset.csv")
    except FileNotFoundError:
        print("No dataset.csv found -> generating synthetic data")
        df = synthetic_data.generate()

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    # 2. Hold out a test set the model never sees during training
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 3. Train
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )),
    ])
    clf.fit(X_tr, y_tr)

    # 4. Evaluate
    pred = clf.predict(X_te)
    proba = clf.predict_proba(X_te)[:, 1]
    print("\n=== Evaluation on held-out test set ===")
    print(classification_report(y_te, pred, target_names=["benign", "malware"]))
    print("Confusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(y_te, pred))
    print(f"ROC-AUC: {roc_auc_score(y_te, proba):.4f}")

    # False-positive / false-negative rates matter most in security
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    print(f"False positive rate (clean flagged as malware): {fp/(fp+tn):.4f}")
    print(f"False negative rate (malware missed):           {fn/(fn+tp):.4f}")

    # 5. Feature importances -> explainability
    rf = clf.named_steps["rf"]
    imp = sorted(zip(FEATURE_NAMES, rf.feature_importances_),
                 key=lambda x: x[1], reverse=True)
    print("\nTop signals the model relies on:")
    for name, score in imp[:8]:
        print(f"  {name:<22} {score:.3f}")

    # 6. Save
    joblib.dump({"pipeline": clf, "features": FEATURE_NAMES}, model_path)
    print(f"\nSaved model -> {model_path}")


if __name__ == "__main__":
    main()
