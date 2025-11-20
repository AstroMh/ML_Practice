import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – safe in all environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)


# -------------------------------------------------------------------
# Paths & data loading
# -------------------------------------------------------------------

def get_project_paths(results_dir_name: str = "results"):
    """
    Helper to get absolute paths based on the location of this script.
    Dataset is loaded from sklearn (no CSV needed), but we still
    create a results directory next to this script.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, results_dir_name)
    os.makedirs(results_dir, exist_ok=True)
    return base_dir, results_dir


def load_data(as_frame: bool = True):
    """
    Load the Breast Cancer dataset as a pandas DataFrame.
    Returns both the DataFrame and the original sklearn Bunch object.

    DataFrame includes all 30 features and 'target' column:
      - 0 = malignant
      - 1 = benign
    """
    data = load_breast_cancer(as_frame=as_frame)
    df = data.frame
    return df, data


# -------------------------------------------------------------------
# Exploratory analysis & preprocessing
# -------------------------------------------------------------------

def explore_data(df: pd.DataFrame, results_dir: str) -> None:
    """
    Basic exploratory visualization.

    Saves:
      - pairplot of a subset of features + target
      - correlation heatmap of all numeric columns
    """
    # Subset for pairplot (to keep it readable)
    cols_for_pairplot = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness",
        "target",
    ]
    subset = df[cols_for_pairplot].copy()

    # Pairplot
    pairplot_path = os.path.join(results_dir, "breast_cancer_rf_pairplot.png")
    sns.pairplot(subset, hue="target")
    plt.tight_layout()
    plt.savefig(pairplot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Pairplot saved to {pairplot_path}")

    # Correlation heatmap (all numeric cols, including target)
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Breast Cancer – Correlation Heatmap")
    plt.tight_layout()
    heatmap_path = os.path.join(results_dir, "breast_cancer_rf_corr_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Correlation heatmap saved to {heatmap_path}")


def prepare_features(df: pd.DataFrame):
    """
    Prepare feature matrix X and target vector y.

    The DataFrame from load_breast_cancer(as_frame=True) contains:
      - 30 numeric feature columns
      - 'target' column (0 = malignant, 1 = benign)
    """
    if "target" not in df.columns:
        raise ValueError("Expected 'target' column not found in DataFrame.")

    X = df.drop("target", axis=1)
    y = df["target"].astype(int)
    feature_names = X.columns.tolist()
    return X, y, feature_names


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split features and target into train and test sets with stratification.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=y,
    )


# -------------------------------------------------------------------
# Modeling & evaluation
# -------------------------------------------------------------------

def train_random_forest(
    X_train,
    y_train,
    n_estimators: int = 500,
    max_depth=None,
    random_state: int = 42,
):
    """
    Fit a Random Forest classifier.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_classifier(
    model,
    X_test,
    y_test,
    results_dir: str,
    report_filename: str = "breast_cancer_random_forest_report.txt",
    cm_filename: str = "breast_cancer_random_forest_confusion_matrix.png",
):
    """
    Evaluate a classification model and save metrics + confusion matrix.

    Saves:
      - Text report (accuracy + classification report + confusion matrix)
      - Confusion matrix heatmap as PNG
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    # Print to console
    print("\n=== Breast Cancer – Random Forest Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(cr)

    # Save text report
    report_path = os.path.join(results_dir, report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Breast Cancer – Random Forest Evaluation\n\n")
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification report:\n")
        f.write(cr)

    print(f"[INFO] Evaluation report saved to {report_path}")

    # Save confusion matrix heatmap
    cm_path = os.path.join(results_dir, cm_filename)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix – Breast Cancer Random Forest")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix heatmap saved to {cm_path}")

    return y_pred, {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": cr,
    }


def plot_roc_curve(
    model,
    X_test,
    y_test,
    results_dir: str,
    filename: str = "breast_cancer_random_forest_roc_curve.png",
):
    """
    Plot ROC curve and save it as PNG. Also returns FPR, TPR, AUC.
    """
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Model does not support predict_proba; cannot compute ROC curve.")

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Breast Cancer Random Forest")
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_path = os.path.join(results_dir, filename)
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] ROC curve saved to {roc_path}")

    return fpr, tpr, roc_auc


def plot_feature_importances(
    model,
    feature_names,
    results_dir: str,
    filename: str = "breast_cancer_random_forest_feature_importances.png",
):
    """
    Plot feature importances as a horizontal bar chart and save to PNG.
    """
    importances = model.feature_importances_
    series = pd.Series(importances, index=feature_names).sort_values()

    if len(series) == 0:
        print("[WARN] No feature importances to plot.")
        return

    height = max(4, len(series) * 0.25)
    plt.figure(figsize=(8, height))
    series.plot(kind="barh")
    plt.xlabel("Feature importance")
    plt.title("Random Forest Feature Importances – Breast Cancer")
    plt.tight_layout()

    fi_path = os.path.join(results_dir, filename)
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Feature importances plot saved to {fi_path}")


def save_feature_importances_table(
    model,
    feature_names,
    results_dir: str,
    filename_csv: str = "breast_cancer_random_forest_feature_importances.csv",
    filename_txt: str = "breast_cancer_random_forest_feature_importances.txt",
):
    """
    Save feature importances to CSV and TXT files.
    """
    importances = model.feature_importances_
    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    csv_path = os.path.join(results_dir, filename_csv)
    fi_df.to_csv(csv_path, index=False)

    txt_path = os.path.join(results_dir, filename_txt)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Breast Cancer – Random Forest Feature Importances\n\n")
        for _, row in fi_df.iterrows():
            f.write(f"{row['feature']}: {row['importance']:.6f}\n")

    print(f"[INFO] Feature importances saved to {csv_path} and {txt_path}")


# -------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------

def main():
    # Resolve paths
    _, results_dir = get_project_paths(results_dir_name="results")

    # 1. Load data
    df, data_bunch = load_data(as_frame=True)

    # 2. Exploratory analysis
    explore_data(df, results_dir=results_dir)

    # 3. Prepare features and target
    X, y, feature_names = prepare_features(df)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Train Random Forest
    rf_model = train_random_forest(
        X_train,
        y_train,
        n_estimators=500,
        max_depth=None,
    )

    # 6. Evaluate model
    y_pred, metrics = evaluate_classifier(
        rf_model,
        X_test,
        y_test,
        results_dir=results_dir,
        report_filename="breast_cancer_random_forest_report.txt",
        cm_filename="breast_cancer_random_forest_confusion_matrix.png",
    )

    # 7. ROC curve
    plot_roc_curve(
        rf_model,
        X_test,
        y_test,
        results_dir=results_dir,
        filename="breast_cancer_random_forest_roc_curve.png",
    )

    # 8. Feature importances (plots + tables)
    plot_feature_importances(
        rf_model,
        feature_names,
        results_dir=results_dir,
        filename="breast_cancer_random_forest_feature_importances.png",
    )
    save_feature_importances_table(
        rf_model,
        feature_names,
        results_dir=results_dir,
        filename_csv="breast_cancer_random_forest_feature_importances.csv",
        filename_txt="breast_cancer_random_forest_feature_importances.txt",
    )


if __name__ == "__main__":
    main()
