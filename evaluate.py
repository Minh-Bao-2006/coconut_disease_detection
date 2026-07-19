import argparse, numpy as np, joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score,
    classification_report, confusion_matrix)
from config import *

def evaluate(feat_path, clf_path):
    d    = np.load(feat_path)
    pipe = joblib.load(clf_path)
    X_te, y_te = d["X_test"], d["y_test"]

    preds = pipe.predict(X_te)
    probs = pipe.predict_proba(X_te)
    acc   = accuracy_score(y_te, preds)
    f1    = f1_score(y_te, preds, average="macro")

    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1 : {f1:.4f}")
    print("\n" + classification_report(y_te, preds, target_names=CLASSES))

    Path(RESULTS_DIR).mkdir(exist_ok=
True
)
    with open(f"{RESULTS_DIR}/report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nF1: {f1:.4f}\n\n")
        f.write(classification_report(y_te, preds, target_names=CLASSES))

    cm = confusion_matrix(y_te, preds)
    fig, ax = plt.subplots(figsize=(
8
, 
6
))
    sns.heatmap(cm, annot=
True
, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f"Confusion Matrix — Acc={acc:.4f}")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=
150
)
    print(f"Saved: {RESULTS_DIR}/confusion_matrix.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=f"{FEATURES_DIR}/features_{BACKBONE}.npz")
    ap.add_argument("--clf",  default=f"{CHECKPOINTS}/svm_pipeline.pkl")
    args = ap.parse_args()
    evaluate(args.feat, args.clf)