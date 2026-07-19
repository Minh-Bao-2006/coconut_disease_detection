import numpy as np
import joblib
import wandb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

CLASSES = ["Bud Root Dropping", "Bud Rot",
           "Gray Leaf Spot", "Leaf Rot", "Stem Bleeding"]

def load_features(path="features/features_resnet18.npz"):
    d = np.load(path)
    X_train = np.concatenate([d["X_train"], d["X_val"]])
    y_train = np.concatenate([d["y_train"], d["y_val"]])
    return X_train, y_train, d["X_test"], d["y_test"]

SVM_ABLATION_GRID = [
    {"pca": 128,  "C": 1},
    {"pca": 128,  "C": 10},
    {"pca": 256,  "C": 1},
    {"pca": 256,  "C": 10},
    {"pca": 512,  "C": 10},
    {"pca": None, "C": 10},  
]

def build_svm_pipeline(pca_dim=256, C=10):
    steps = [("scaler", StandardScaler())]
    if pca_dim is not None:
        steps.append(("pca", PCA(n_components=pca_dim, whiten=True)))
    steps.append(("svm", SVC(kernel="rbf", C=C, gamma="scale",
                              probability=True, random_state=42)))
    return Pipeline(steps)

def build_rf_pipeline(pca_dim=200):
    steps = [("scaler", StandardScaler())]
    if pca_dim is not None:
        steps.append(("pca", PCA(n_components=pca_dim)))
    steps.append(("rf", RandomForestClassifier(
        n_estimators=500, max_depth=25,
        min_samples_leaf=2, n_jobs=-1, random_state=42
    )))
    return Pipeline(steps)

def run_svm_ablation(X_train, y_train, X_val, y_val, run):
    print("\n" + "="*50)
    print("ABLATION STUDY: PCA Dims × SVM-C (ResNet-18 + SVM)")
    print("="*50)

    table = wandb.Table(columns=["PCA Dims", "SVM C", "Val Acc (%)", "Test Acc (%)"])
    results = []

    for cfg in SVM_ABLATION_GRID:
        pca_dim = cfg["pca"]
        C       = cfg["C"]
        label   = str(pca_dim) if pca_dim else "None (no PCA)"

        pipe = build_svm_pipeline(pca_dim=pca_dim, C=C)
        pipe.fit(X_train, y_train)

        val_acc  = accuracy_score(y_train, pipe.predict(X_train))  
        test_acc = accuracy_score(y_val,   pipe.predict(X_val))

        val_pct  = round(val_acc  * 100, 2)
        test_pct = round(test_acc * 100, 2)

        print(f"  PCA={label:>12s} | C={C:>4} | Val={val_pct:.2f}% | Test={test_pct:.2f}%")
        table.add_data(label, C, val_pct, test_pct)
        results.append({
            "pca": pca_dim, "C": C,
            "val_acc": val_acc, "test_acc": test_acc,
            "pipe": pipe
        })

        wandb.log({
            "SVM_Ablation/pca_dim": pca_dim if pca_dim else 0,
            "SVM_Ablation/C":       C,
            "SVM_Ablation/val_acc": val_pct,
            "SVM_Ablation/test_acc":test_pct,
        })

    wandb.log({"Table_IV_SVM_Ablation": table})

    best = max(results, key=lambda x: x["test_acc"])
    print(f"\n  ✓ Best SVM config: PCA={best['pca']}, C={best['C']} "
          f"→ Test Acc={best['test_acc']*100:.2f}%")
    return best["pipe"], best

def log_confusion_matrix(prefix, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    n  = len(CLASSES)

    data = []
    for i in range(n):
        for j in range(n):
            data.append([CLASSES[i], CLASSES[j], int(cm[i][j])])
    table = wandb.Table(columns=["Actual", "Predicted", "Count"], data=data)
    wandb.log({
        f"{prefix}/confusion_matrix": wandb.plot.bar(
            table, "Predicted", "Count",
            title=f"{prefix} Confusion Matrix"
        )
    })

    cm_table = wandb.Table(
        columns=[""] + CLASSES,
        data=[[CLASSES[i]] + [int(cm[i][j]) for j in range(n)] for i in range(n)]
    )
    wandb.log({f"{prefix}/confusion_matrix_table": cm_table})

def log_model_metrics(run, prefix, pipe, X_train, y_train, X_test, y_test, model_type):
    train_preds = pipe.predict(X_train)
    test_preds  = pipe.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc  = accuracy_score(y_test,  test_preds)
    report    = classification_report(y_test, test_preds,
                                      target_names=CLASSES, output_dict=True)

    print(f"\n[{prefix}] Train Accuracy : {train_acc*100:.2f}%")
    print(f"[{prefix}] Test  Accuracy : {test_acc*100:.2f}%")
    print(classification_report(y_test, test_preds, target_names=CLASSES))

    wandb.log({
        f"{prefix}/train_accuracy": train_acc,
        f"{prefix}/test_accuracy":  test_acc,
    })

    for cls in CLASSES:
        wandb.log({
            f"{prefix}/{cls}/precision": report[cls]["precision"],
            f"{prefix}/{cls}/recall":    report[cls]["recall"],
            f"{prefix}/{cls}/f1-score":  report[cls]["f1-score"],
            f"{prefix}/{cls}/support":   report[cls]["support"],
        })

    for avg in ["macro avg", "weighted avg"]:
        key = avg.replace(" ", "_")
        wandb.log({
            f"{prefix}/{key}/precision": report[avg]["precision"],
            f"{prefix}/{key}/recall":    report[avg]["recall"],
            f"{prefix}/{key}/f1-score":  report[avg]["f1-score"],
        })

    log_confusion_matrix(prefix, y_test, test_preds)

    model_path = f"checkpoints/{model_type}_pipeline.pkl"
    joblib.dump(pipe, model_path)
    artifact = wandb.Artifact(
        name=f"{model_type}_pipeline", type="model",
        description=f"{prefix} pipeline trained on ResNet18 features",
    )
    artifact.add_file(model_path)
    run.log_artifact(artifact)


def train_all(feat_path="features/features_resnet18.npz"):
    X_train, y_train, X_test, y_test = load_features(feat_path)

    wandb.login(key="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")
    if wandb.run is not None:
        wandb.finish()

    run = wandb.init(
        project="coconut-disease-detection",
        entity="baonguyenminh17-fpt-university",
        name="SVM-RF-ResNet18-Features-v4",   
        config={
            "SVM/kernel": "rbf", "SVM/C": 10, "SVM/gamma": "scale",
            "SVM/pca_components": 256, "SVM/pca_whiten": True,
            "RF/n_estimators": 500, "RF/max_depth": 25,
            "RF/min_samples_leaf": 2, "RF/pca_components": 200,
            "feature_extractor": "ResNet18",
        }
    )

    best_svm_pipe, best_cfg = run_svm_ablation(X_train, y_train, X_test, y_test, run)

    print("\n" + "="*40)
    print("Training BEST SVM...")
    svm_pipe = build_svm_pipeline(pca_dim=best_cfg["pca"], C=best_cfg["C"])
    svm_pipe.fit(X_train, y_train)
    log_model_metrics(run, "SVM", svm_pipe, X_train, y_train, X_test, y_test, "svm")

    print("="*40)
    print("Training Random Forest...")
    rf_pipe = build_rf_pipeline(pca_dim=200)
    rf_pipe.fit(X_train, y_train)
    log_model_metrics(run, "RF", rf_pipe, X_train, y_train, X_test, y_test, "rf")

    wandb.finish()
    print("\nDone! Ablation + SVM + RF are log in Wandb.")
    return svm_pipe, rf_pipe


if __name__ == "__main__":
    train_all("features/features_resnet18.npz")