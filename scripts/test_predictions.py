"""
Test the prediction pipeline with multiple input cases.
"""

import sys, os, warnings
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

from predict import load_model, predict
from feature_engineering import create_single_features, FEATURE_COLS

model, stats = load_model()

print("Model type:", type(model).__name__)
print("model.classes_:", model.classes_)
print()

TEST_CASES = [
    {
        "label": "TEST 1 — Normal (Paracetamol, Rs 150, B12345, 2yr shelf)",
        "form": {
            "medicine_name"   : "Paracetamol",
            "manufacturer"    : "Sun Pharma",
            "batch_number"    : "B12345",
            "country"         : "India",
            "manufacture_date": "2026-01-15",
            "expiry_date"     : "2028-01-15",
            "dosage"          : 500,
            "price"           : 150,
        }
    },
    {
        "label": "TEST 2 — Suspicious (Paracetamol, Rs 10, Batch A1, 31-day shelf)",
        "form": {
            "medicine_name"   : "Paracetamol",
            "manufacturer"    : "Sun Pharma",
            "batch_number"    : "A1",
            "country"         : "India",
            "manufacture_date": "2026-07-01",
            "expiry_date"     : "2026-08-01",
            "dosage"          : 500,
            "price"           : 10,
        }
    },
    {
        "label": "TEST 3 — High price (Paracetamol Rs 900, normal batch/shelf)",
        "form": {
            "medicine_name"   : "Paracetamol",
            "manufacturer"    : "Cipla",
            "batch_number"    : "B789012",
            "country"         : "India",
            "manufacture_date": "2025-06-01",
            "expiry_date"     : "2027-06-01",
            "dosage"          : 500,
            "price"           : 900,
        }
    },
    {
        "label": "TEST 4 — Rare manufacturer, short batch, low price",
        "form": {
            "medicine_name"   : "Amoxicillin",
            "manufacturer"    : "Vedic Pharma",
            "batch_number"    : "RX123",
            "country"         : "India",
            "manufacture_date": "2026-05-01",
            "expiry_date"     : "2026-08-01",
            "dosage"          : 250,
            "price"           : 45,
        }
    },
    {
        "label": "TEST 5 — Near-expiry but normal price and shelf life",
        "form": {
            "medicine_name"   : "Azithromycin",
            "manufacturer"    : "Dr. Reddy's",
            "batch_number"    : "B567890",
            "country"         : "Germany",
            "manufacture_date": "2024-01-01",
            "expiry_date"     : "2026-08-15",
            "dosage"          : 500,
            "price"           : 290,
        }
    },
    {
        "label": "TEST 6 — Unknown manufacturer, very short shelf, very low price",
        "form": {
            "medicine_name"   : "Ibuprofen",
            "manufacturer"    : "QuickMed Labs",
            "batch_number"    : "CF99",
            "country"         : "India",
            "manufacture_date": "2026-07-01",
            "expiry_date"     : "2026-09-01",
            "dosage"          : 400,
            "price"           : 20,
        }
    },
]

for tc in TEST_CASES:
    print("=" * 68)
    print(tc["label"])
    print("=" * 68)

    form = tc["form"]
    risk, p_cf, factors = predict(form, model, stats)

    # Also print raw features
    feat_raw = create_single_features(form, {**stats, "scaler": None})
    print()
    print("  Engineered features (before scaling):")
    for name, val in zip(FEATURE_COLS, feat_raw[0]):
        print(f"    {name:<26}: {val:.3f}")

    print()
    print(f"  Estimated Counterfeit Probability : {p_cf:.1f}%")
    print(f"  Risk Level                        : {risk}")
    print()
    print("  Observed Risk Factors:")
    for f in factors:
        print(f"    - {f}")
    print()
