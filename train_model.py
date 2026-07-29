"""
Train the Counterfeit Medicine Risk Prediction Model.

Steps:
1. Load and clean the dataset
2. Split into train / test FIRST (stratified)
3. Compute feature-engineering statistics from training data ONLY
4. Create features for both train and test using those statistics
5. Fit a StandardScaler on training features (saved with model)
6. Train Logistic Regression, Decision Tree, Random Forest
7. Select the best model by Counterfeit-class F1 score
8. Evaluate and display full classification report + confusion matrix
9. Save the model, stats (including scaler) to a single .pkl file

IMPORTANT:
- Statistics (avg price, country risk, manufacturer frequency) are
  computed from the training split ONLY to prevent data leakage.
- The scaler is fit on training features only and saved with the model
  so inference (Flask prediction) uses identical preprocessing.
- Model selection uses Counterfeit F1, not overall accuracy,
  because missing a counterfeit medicine is the critical error.
"""

import os
import pickle
import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from config import DATA_PATH, MODEL_PATH
from feature_engineering import (
    FEATURE_COLS,
    compute_stats_from_train,
    create_features,
)


# ── 1. Load ────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load the dataset."""
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} records.")
    return df


# ── 2. Clean ───────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid records."""

    df = df.drop_duplicates()

    df["manufacturer"] = df["manufacturer"].astype(str).str.strip().str.strip(',"')

    df["manufacture_date"] = pd.to_datetime(df["manufacture_date"], errors="coerce")
    df["expiry_date"]      = pd.to_datetime(df["expiry_date"],      errors="coerce")

    df = df.dropna(subset=["manufacture_date", "expiry_date"])
    df = df[df["expiry_date"] > df["manufacture_date"]]
    df = df[df["price_inr"] > 0]
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    print(f"  Records after cleaning: {len(df)}")
    return df


# ── 3. Train models ────────────────────────────────────────────────────────────

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Train Logistic Regression, Decision Tree, and Random Forest.

    Note: X_train and X_test passed here are ALREADY scaled by StandardScaler
    for Logistic Regression. Decision Tree / Random Forest are tree-based and
    do not require scaling, but applying it does not harm them.

    Model selection criterion: F1-score for the Counterfeit class (label=0),
    because missing a counterfeit medicine is the critical error.
    """

    candidates = [
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ("Decision Tree",
         DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42)),
        ("Random Forest",
         RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                max_depth=10, random_state=42)),
    ]

    print(f"\n  {'Model':<25} {'Accuracy':>9} {'CF-Prec':>9} {'CF-Rec':>8} {'CF-F1':>8}")
    print(f"  {'-'*25} {'-'*9} {'-'*9} {'-'*8} {'-'*8}")

    best_model      = None
    best_cf_f1      = -1.0
    best_name       = ""
    results         = []

    for name, model in candidates:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc     = accuracy_score(y_test, y_pred)
        cf_f1   = f1_score(y_test, y_pred, pos_label=0)
        cf_prec = 0.0
        cf_rec  = 0.0

        # Extract per-class metrics safely
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        if "0" in report:
            cf_prec = report["0"]["precision"]
            cf_rec  = report["0"]["recall"]

        print(f"  {name:<25} {acc:>9.4f} {cf_prec:>9.4f} {cf_rec:>8.4f} {cf_f1:>8.4f}")
        results.append((name, model, acc, cf_prec, cf_rec, cf_f1))

        if cf_f1 > best_cf_f1:
            best_cf_f1  = cf_f1
            best_model  = model
            best_name   = name

    print(f"\n  Selected: {best_name}  (highest Counterfeit F1 = {best_cf_f1:.4f})")
    return best_model, best_name, results


# ── 4. Evaluate selected model ─────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str):
    """Print full evaluation for the selected model."""

    y_pred = model.predict(X_test)

    print(f"\n  === Full Evaluation: {model_name} ===")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=["Counterfeit (0)", "Genuine (1)"],
        zero_division=0,
    ))

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"                       Pred Counterfeit   Pred Genuine")
    print(f"  Actual Counterfeit        {cm[0,0]:9d}      {cm[0,1]:9d}")
    print(f"  Actual Genuine            {cm[1,0]:9d}      {cm[1,1]:9d}")

    # Show feature importance or coefficients
    print()
    if hasattr(model, "feature_importances_"):
        print("  Feature Importances:")
        pairs = sorted(zip(FEATURE_COLS, model.feature_importances_),
                       key=lambda x: -x[1])
        for feat, imp in pairs:
            bar = "#" * int(imp * 40)
            print(f"    {feat:<26} {imp:.4f}  {bar}")

    elif hasattr(model, "coef_"):
        print("  Logistic Regression Coefficients (after scaling):")
        print("  Positive -> Genuine, Negative -> Counterfeit")
        pairs = sorted(zip(FEATURE_COLS, model.coef_[0]),
                       key=lambda x: -abs(x[1]))
        for feat, coef in pairs:
            direction = "Genuine" if coef > 0 else "Counterfeit"
            print(f"    {feat:<26} {coef:+.4f}  -> {direction}")


# ── 5. Save ────────────────────────────────────────────────────────────────────

def save_model(model, stats: dict) -> None:
    """Save the trained model and all preprocessing stats."""

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    payload = {
        "model" : model,
        "stats" : stats,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
    print(f"\n  Model saved to  : {MODEL_PATH}")
    print(f"  Last modified   : {mtime}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\nCounterfeit Medicine Model Training\n")

    # ── Load and clean ─────────────────────────────────────────────────────────
    print("Step 1: Loading dataset...")
    df = load_data()

    print("Step 2: Cleaning dataset...")
    df = clean_data(df)

    print(f"\n  Label distribution:")
    vc = df["label"].value_counts()
    print(f"    Genuine     (1): {vc.get(1, 0)}")
    print(f"    Counterfeit (0): {vc.get(0, 0)}")
    print(f"    Counterfeit rate: {vc.get(0,0)/len(df)*100:.1f}%")

    # ── Split FIRST ────────────────────────────────────────────────────────────
    # We split on raw (pre-feature-engineering) data so stats can be computed
    # from the training split only.
    print("\nStep 3: Splitting dataset (stratified, 80/20)...")
    df_train, df_test = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=42,
    )
    print(f"  Training records : {len(df_train)}")
    print(f"  Test records     : {len(df_test)}")

    # ── Compute stats from TRAINING data only ──────────────────────────────────
    print("\nStep 4: Computing feature statistics from training data only...")
    stats = compute_stats_from_train(df_train.copy())

    # ── Create features ────────────────────────────────────────────────────────
    print("Step 5: Creating features...")
    df_train = create_features(df_train.copy(), stats)
    df_test  = create_features(df_test.copy(),  stats)

    X_train = df_train[FEATURE_COLS].fillna(0).values
    y_train = df_train["label"].values

    X_test  = df_test[FEATURE_COLS].fillna(0).values
    y_test  = df_test["label"].values

    # ── Scale features ─────────────────────────────────────────────────────────
    # Fit on TRAINING data only; transform both train and test.
    # The scaler is saved in stats so inference uses the same scaling.
    print("Step 6: Fitting StandardScaler on training features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Store scaler in stats so it can be used at inference time
    stats["scaler"] = scaler

    # ── Train and select model ─────────────────────────────────────────────────
    print("\nStep 7: Training and comparing models...")
    best_model, best_name, _ = train_and_evaluate(
        X_train_scaled, y_train,
        X_test_scaled,  y_test,
    )

    # ── Evaluate selected model ────────────────────────────────────────────────
    print("\nStep 8: Evaluating selected model...")
    evaluate_model(best_model, X_test_scaled, y_test, best_name)

    # ── Save ───────────────────────────────────────────────────────────────────
    print("\nStep 9: Saving model and stats...")
    save_model(best_model, stats)

    print("\nTraining complete. Run 'python app.py' to start the application.")


if __name__ == "__main__":
    main()