import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier

import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Ensure plots use a nice style
sns.set(style="whitegrid")

# Paths for saving model and plots
MODEL_PATH = "churn_model.pkl"
STATIC_DIR = "static"
SHAP_SUMMARY_PLOT_PATH = os.path.join(STATIC_DIR, "shap_summary.png")
CHURN_DISTRIBUTION_PLOT_PATH = os.path.join(STATIC_DIR, "churn_distribution.png")
TENURE_CHURN_PLOT_PATH = os.path.join(STATIC_DIR, "tenure_churn.png")
MONTHLY_CHARGES_CHURN_PLOT_PATH = os.path.join(STATIC_DIR, "monthly_charges_churn.png")

def load_dataset_from_csv(file_path: str, target_col: str = "Churn"):
    df = pd.read_csv(file_path)

    # ✅ DROP NON-USEFUL COLUMNS (ADD HERE)
    if "Surname" in df.columns:
        df = df.drop(columns=["Surname"])

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. "
            f"Columns found: {list(df.columns)}"
        )

    y_raw = df[target_col]

    if y_raw.dtype == "object":
        y = y_raw.map({"Yes": 1, "No": 0})
    else:
        y = y_raw

    mask_notna = ~y.isna()
    df = df.loc[mask_notna].copy()
    y = y.loc[mask_notna]

    X = df.drop(columns=[target_col])
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    return X, y



def build_preprocessor(X: pd.DataFrame):
    """
    Build a ColumnTransformer preprocessor:
    - numeric features: passthrough
    - categorical features: OneHotEncoder
    """
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Perform train_test_split.
    If any class has fewer than 2 samples, do NOT use stratify.
    """
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("Class distribution:", class_counts)

    if len(class_counts) < 2 or min(class_counts.values()) < 2:
        print(
            "WARNING: One or more classes have fewer than 2 samples. "
            "Proceeding without stratify. Evaluation metrics may be unreliable."
        )
        stratify_arg = None
    else:
        stratify_arg = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )
    return X_train, X_test, y_train, y_test


def train_churn_model(csv_path: str, target_col: str = "Churn"):
    """
    Train churn model from a CSV file and generate plots.
    """
    os.makedirs(STATIC_DIR, exist_ok=True)

    # Load data
    X, y = load_dataset_from_csv(csv_path, target_col=target_col)

    # Build preprocessor
    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # Class weights
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            f"Target column '{target_col}' has only one class: {classes}. "
            f"You need at least two classes (e.g., 0 and 1) to train a classifier."
        )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )
    class_weight_dict = dict(zip(classes, class_weights))
    print("Computed class weights:", class_weight_dict)

    scale_pos_weight = class_weight_dict.get(1, 1.0)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = safe_train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    try:
        roc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        roc = float("nan")

    try:
        f1 = f1_score(y_test, y_pred)
    except ValueError:
        f1 = float("nan")

    print(f"ROC-AUC: {roc}, F1-score: {f1}")

    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Generate visualizations
    try:
        create_basic_charts(X_train, y_train)
    except Exception as e:
        print("Error generating basic charts:", e)

    try:
        shap_summary_plot(pipeline, X_train)
    except Exception as e:
        print("Error generating SHAP plot:", e)

    return {
        "roc_auc": float(roc) if not np.isnan(roc) else None,
        "f1": float(f1) if not np.isnan(f1) else None,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


def create_basic_charts(X_train, y_train):
    """
    Create simple, easy-to-understand charts:
    - Churn distribution
    - Tenure vs churn (boxplot)
    - MonthlyCharges vs churn (boxplot)
    """
    df = X_train.copy()
    df["churn"] = y_train.values

    # Churn distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(x="churn", data=df, palette="Set2")
    plt.title("Churn Distribution (0 = Stayed, 1 = Churned)")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(CHURN_DISTRIBUTION_PLOT_PATH, dpi=120)
    plt.close()

    # Tenure vs churn
    if "tenure" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="churn", y="tenure", data=df, palette="Set3")
        plt.title("Tenure vs Churn")
        plt.xlabel("Churn")
        plt.ylabel("Tenure (months)")
        plt.tight_layout()
        plt.savefig(TENURE_CHURN_PLOT_PATH, dpi=120)
        plt.close()

    # MonthlyCharges vs churn
    # handle Telco naming: MonthlyCharges / monthly_charges
    monthly_col = None
    for c in df.columns:
        if c.lower() in ["monthlycharges", "monthly_charges"]:
            monthly_col = c
            break

    if monthly_col:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="churn", y=monthly_col, data=df, palette="Pastel1")
        plt.title("Monthly Charges vs Churn")
        plt.xlabel("Churn")
        plt.ylabel("Monthly Charges")
        plt.tight_layout()
        plt.savefig(MONTHLY_CHARGES_CHURN_PLOT_PATH, dpi=120)
        plt.close()


def shap_summary_plot(pipeline, X_train, max_samples=500):
    """
    Generate a SHAP summary plot for the trained pipeline and save it as an image.
    """
    os.makedirs(STATIC_DIR, exist_ok=True)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Sample for SHAP
    if len(X_train) > max_samples:
        X_sample = X_train.sample(max_samples, random_state=42)
    else:
        X_sample = X_train.copy()

    X_processed = preprocessor.transform(X_sample)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_processed)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()


def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict_single_customer(input_dict: dict):
    model = load_trained_model()
    if model is None:
        raise RuntimeError("Model not trained yet. Please train the model first.")

    X = pd.DataFrame([input_dict])
    proba = model.predict_proba(X)[0, 1]
    label = int(proba >= 0.5)

    return {
        "probability": float(proba),
        "label": label,
    }


__all__ = [
    "train_churn_model",
    "predict_single_customer",
    "SHAP_SUMMARY_PLOT_PATH",
    "CHURN_DISTRIBUTION_PLOT_PATH",
    "TENURE_CHURN_PLOT_PATH",
    "MONTHLY_CHARGES_CHURN_PLOT_PATH",
    "load_trained_model",
]
