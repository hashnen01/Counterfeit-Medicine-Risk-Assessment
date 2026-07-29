"""
Prediction functions for the Counterfeit Medicine Risk Assessment System.

Key design decisions:
- P(Counterfeit) is extracted using model.classes_ so the correct
  probability index is always used regardless of class ordering.
- Risk level is derived from P(Counterfeit), not from max(probabilities).
- Thresholds are documented below and can be tuned without touching
  the ML model.
- generate_risk_factors() is a separate rule-based function.
  It does NOT explain the ML model — it identifies observable warning
  signals independently. The UI labels this section "Observed Risk Factors"
  to make clear it is not a model explanation.
"""

import pickle
from config import MODEL_PATH
from feature_engineering import create_single_features


# ── Risk thresholds for P(Counterfeit) ────────────────────────────────────────
# These determine Low / Medium / High risk labels.
# They were chosen based on the training dataset's counterfeit rate (~22%)
# and the goal of being sensitive to counterfeit risk.
THRESHOLD_HIGH   = 0.50   # P(Counterfeit) >= 0.50  -> High
THRESHOLD_MEDIUM = 0.25   # P(Counterfeit) >= 0.25  -> Medium
                           # P(Counterfeit) <  0.25  -> Low


def load_model():
    """Load the trained model and stats (including scaler)."""
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["stats"]


def get_counterfeit_probability(model, features) -> float:
    """
    Return P(Counterfeit) for the given feature vector.

    Uses model.classes_ to find the index of class 0 (Counterfeit)
    so the result is correct regardless of class ordering.
    """
    probabilities = model.predict_proba(features)[0]
    classes       = list(model.classes_)

    if 0 not in classes:
        # Fallback: assume lower-index class is counterfeit
        return float(probabilities[0])

    cf_index = classes.index(0)
    return float(probabilities[cf_index])


def probability_to_risk(p_counterfeit: float) -> str:
    """
    Convert P(Counterfeit) to a risk label.

    Thresholds:
        >= 0.50  -> High
        >= 0.25  -> Medium
        <  0.25  -> Low
    """
    if p_counterfeit >= THRESHOLD_HIGH:
        return "High"
    elif p_counterfeit >= THRESHOLD_MEDIUM:
        return "Medium"
    else:
        return "Low"


def generate_risk_factors(form_data: dict, stats: dict) -> list:
    """
    Generate observable risk indicator messages.

    IMPORTANT: These are RULE-BASED checks on the input data.
    They do NOT explain the ML model prediction.
    The UI displays them under "Observed Risk Factors", not "Reasons".
    """
    factors = []

    medicine     = form_data["medicine_name"]
    price        = float(form_data["price"])
    manufacturer = form_data["manufacturer"]
    country      = form_data["country"]
    batch        = str(form_data["batch_number"])

    # 1. Price check
    drug_avg = stats["drug_avg_price"]
    avg_price = drug_avg.get(medicine, 0)
    if avg_price > 0:
        if price < avg_price * 0.50:
            diff = round((1 - price / avg_price) * 100)
            factors.append(
                f"Price (Rs {price:.0f}) is {diff}% below the average market "
                f"price for {medicine} (Rs {avg_price:.0f})."
            )
        elif price > avg_price * 1.80:
            diff = round((price / avg_price - 1) * 100)
            factors.append(
                f"Price (Rs {price:.0f}) is {diff}% above the average market "
                f"price for {medicine} (Rs {avg_price:.0f})."
            )

    # 2. Manufacturer check
    mfr_freq  = stats["mfr_frequency"]
    threshold = stats["mfr_common_threshold"]
    count     = mfr_freq.get(manufacturer, 0)
    if count == 0:
        factors.append(
            f"Manufacturer '{manufacturer}' was not found in training records."
        )
    elif count < threshold:
        factors.append(
            f"Manufacturer '{manufacturer}' has very few records ({count}) "
            f"in the training dataset."
        )

    # 3. Country risk check
    c_risk = stats["country_risk"].get(country, 0.22)
    if c_risk > 0.30:
        factors.append(
            f"Country '{country}' has a relatively higher counterfeit rate "
            f"({c_risk*100:.0f}%) in training data."
        )

    # 4. Batch number check
    if len(batch) <= 3:
        factors.append(
            f"Batch number '{batch}' is very short ({len(batch)} characters). "
            f"Standard batch numbers are typically 6+ characters."
        )

    # 5. Shelf life check (if dates are available)
    try:
        import pandas as pd
        mfr_date = pd.Timestamp(form_data["manufacture_date"])
        exp_date = pd.Timestamp(form_data["expiry_date"])
        shelf_days = (exp_date - mfr_date).days
        if shelf_days < 180:
            factors.append(
                f"Shelf life is only {shelf_days} days, which is unusually short "
                f"for most medicines (typically 1-3 years)."
            )
    except Exception:
        pass

    if not factors:
        factors.append("No specific observable risk factors were detected.")

    return factors


def predict(form_data: dict, model, stats: dict):
    """
    Predict counterfeit risk for a single medicine record.

    Returns:
        risk          : "Low", "Medium", or "High"
        p_counterfeit : float, probability of being counterfeit (0-100)
        risk_factors  : list of observable risk factor messages
    """
    features = create_single_features(form_data, stats)

    p_counterfeit = get_counterfeit_probability(model, features)
    risk          = probability_to_risk(p_counterfeit)
    risk_factors  = generate_risk_factors(form_data, stats)

    # Return probability as a percentage rounded to 1 decimal place
    return risk, round(p_counterfeit * 100, 1), risk_factors
