"""
Feature Engineering for Counterfeit Medicine Risk Assessment.

Design principles:
- All prices are in INR (consistent with the Flask UI).
- Statistics required for feature engineering (avg price per drug,
  country risk, manufacturer frequency) are learned from TRAINING DATA
  only and passed in via a 'stats' dictionary.
  This prevents data leakage from the test set into training features.
- Training and inference use exactly the same feature names, order,
  units, and calculations.
- The scaler (StandardScaler) is also saved in the stats dict so
  inference always uses the same scaling as training.
"""

import numpy as np
import pandas as pd

# ── Feature column order must match between training and inference ─────────────
FEATURE_COLS = [
    "price_ratio",           # user price / avg genuine price for that drug
    "shelf_life",            # expiry - manufacture date in days
    "days_until_expiry",     # expiry - today in days
    "medicine_age",          # today - manufacture date in days
    "batch_length",          # length of the batch number string
    "is_common_manufacturer",# 1 if manufacturer seen >= 10 times in training
    "country_risk",          # fraction of counterfeit records for that country
    "price_per_dose",        # price_inr / dosage_mg
]


# ── Training-time feature creation ────────────────────────────────────────────

def compute_stats_from_train(df_train: pd.DataFrame) -> dict:
    """
    Compute all statistics needed for feature engineering
    from the TRAINING data only.

    These stats are saved with the model and reused at inference time.
    """
    # Average genuine price per drug (using label=1 rows only gives a cleaner
    # baseline, but using all training rows is also acceptable and more robust
    # when training data is small; we use all rows here for robustness)
    drug_avg_price = (
        df_train.groupby("drug_name")["price_inr"]
        .mean()
        .to_dict()
    )

    # Manufacturer frequency in training data
    mfr_frequency = df_train["manufacturer"].value_counts().to_dict()

    # Threshold for "common" manufacturer: seen at least 10 times
    mfr_common_threshold = 10

    # Country risk: fraction of counterfeit (label=0) per country
    # Computed from TRAINING labels only.
    country_risk = (
        df_train.groupby("manufacture_country")["label"]
        .apply(lambda v: (v == 0).mean())  # fraction that are counterfeit
        .to_dict()
    )

    stats = {
        "drug_avg_price"        : drug_avg_price,
        "mfr_frequency"         : mfr_frequency,
        "mfr_common_threshold"  : mfr_common_threshold,
        "country_risk"          : country_risk,
        "known_drugs"           : sorted(df_train["drug_name"].dropna().unique().tolist()),
        "known_manufacturers"   : sorted(df_train["manufacturer"].dropna().unique().tolist()),
        "known_countries"       : sorted(df_train["manufacture_country"].dropna().unique().tolist()),
        # scaler is added later by train_model.py after fitting
        "scaler"                : None,
    }
    return stats


def create_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Create all features for a DataFrame (training or test).

    Uses pre-computed stats from compute_stats_from_train().
    Does NOT recompute any statistic from df — no leakage.
    """
    today = pd.Timestamp.now().normalize()  # date only, no time component

    drug_avg = stats["drug_avg_price"]
    mfr_freq = stats["mfr_frequency"]
    threshold = stats["mfr_common_threshold"]
    c_risk    = stats["country_risk"]

    # price_ratio: user price relative to average genuine price for that drug
    df["price_ratio"] = df.apply(
        lambda row: row["price_inr"] / drug_avg.get(row["drug_name"], row["price_inr"])
        if drug_avg.get(row["drug_name"]) else 1.0,
        axis=1,
    )

    # Date features
    df["shelf_life"]        = (df["expiry_date"] - df["manufacture_date"]).dt.days
    df["days_until_expiry"] = (df["expiry_date"] - today).dt.days
    df["medicine_age"]      = (today - df["manufacture_date"]).dt.days

    # Batch length
    df["batch_length"] = df["batch_number"].astype(str).str.len()

    # is_common_manufacturer: 1 if seen >= threshold times in training
    df["is_common_manufacturer"] = df["manufacturer"].apply(
        lambda m: 1 if mfr_freq.get(m, 0) >= threshold else 0
    )

    # country_risk: fraction of counterfeit for that country in training data
    # Default 0.22 (overall dataset counterfeit rate) for unseen countries
    default_risk = 0.22
    df["country_risk"] = df["manufacture_country"].apply(
        lambda c: c_risk.get(c, default_risk)
    )

    # price_per_dose: price in INR per mg
    df["price_per_dose"] = df["price_inr"] / df["dosage_mg"].replace(0, 1)

    return df


# ── Inference-time feature creation (single record from the Flask form) ────────

def create_single_features(form_data: dict, stats: dict) -> np.ndarray:
    """
    Create the feature vector for a single medicine record from the Flask form.

    Uses the same stats and calculations as create_features() above.
    Returns a 2-D numpy array with shape (1, n_features).

    The scaler (if present in stats) is applied here so inference
    matches training exactly.
    """
    today = pd.Timestamp.now().normalize()

    medicine     = form_data["medicine_name"]
    manufacturer = form_data["manufacturer"]
    country      = form_data["country"]
    price        = float(form_data["price"])
    dosage       = float(form_data.get("dosage", 100)) or 1.0

    manufacture_date = pd.Timestamp(form_data["manufacture_date"])
    expiry_date      = pd.Timestamp(form_data["expiry_date"])

    drug_avg  = stats["drug_avg_price"]
    mfr_freq  = stats["mfr_frequency"]
    threshold = stats["mfr_common_threshold"]
    c_risk    = stats["country_risk"]
    default_risk = 0.22

    avg_price = drug_avg.get(medicine, price)
    price_ratio = price / avg_price if avg_price else 1.0

    shelf_life        = (expiry_date - manufacture_date).days
    days_until_expiry = (expiry_date - today).days
    medicine_age      = (today - manufacture_date).days
    batch_length      = len(str(form_data["batch_number"]))
    is_common_mfr     = 1 if mfr_freq.get(manufacturer, 0) >= threshold else 0
    country_risk_val  = c_risk.get(country, default_risk)
    price_per_dose    = price / dosage

    # Feature vector — ORDER must match FEATURE_COLS exactly
    features = np.array([[
        price_ratio,
        shelf_life,
        days_until_expiry,
        medicine_age,
        batch_length,
        is_common_mfr,
        country_risk_val,
        price_per_dose,
    ]])

    # Apply the same scaler used during training
    scaler = stats.get("scaler")
    if scaler is not None:
        features = scaler.transform(features)

    return features
