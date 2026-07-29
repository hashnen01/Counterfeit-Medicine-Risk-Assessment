"""
Generate a synthetic dataset for Counterfeit Medicine Risk Assessment.

Design principles:
- Prices in INR (consistent with the Flask UI).
- Labels are assigned probabilistically, NOT deterministically from features.
- Counterfeit and Genuine records have overlapping but distinct statistical distributions.
- No 'Fake_' in manufacturer names. No 'XXX' batch prefix.
- Realistic noise so the ML problem is non-trivial.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
N_ROWS        = 12000
COUNTERFEIT_RATE = 0.22        # 22% counterfeit is realistic
OUTPUT_PATH   = os.path.join(os.path.dirname(__file__), "data", "medicine_dataset.csv")

# ── Drug catalogue: (name, dosage_mg, baseline_inr) ───────────────────────────
# baseline_inr is the typical genuine price in INR
DRUGS = [
    ("Paracetamol",    500,  120),
    ("Amoxicillin",    250,  180),
    ("Azithromycin",   500,  300),
    ("Omeprazole",      20,  150),
    ("Atorvastatin",    10,  220),
    ("Cetirizine",      10,   80),
    ("Ibuprofen",      400,  100),
    ("Metformin",      500,  130),
    ("Amlodipine",       5,  170),
    ("Pantoprazole",    40,  160),
]

# ── Manufacturers: (name, tier)  ──────────────────────────────────────────────
# tier 1 = very common/reputable, tier 2 = moderately common, tier 3 = rare
MANUFACTURERS = [
    # Tier 1 - large, well-known
    ("Sun Pharma",     1), ("Cipla",           1), ("Dr. Reddy's",   1),
    ("Lupin",          1), ("Zydus Cadila",    1), ("Abbott India",  1),
    ("Pfizer",         1), ("GSK",             1), ("Novartis",      1),
    # Tier 2 - medium
    ("Torrent",        2), ("Mankind Pharma",  2), ("IPCA Labs",     2),
    ("Alkem",          2), ("Glenmark",        2), ("Wockhardt",     2),
    ("Sanofi India",   2), ("Bayer India",     2),
    # Tier 3 - smaller, less common
    ("MedRite Pharma", 3), ("GenPharm Ltd",    3), ("UniPharma",     3),
    ("BioMed Generics",3), ("PharmTech India", 3), ("Vedic Pharma",  3),
    ("Alpha Drugs",    3), ("Nexus Biotech",   3),
]

COUNTRIES = ["India", "USA", "UK", "Germany", "Switzerland"]

# Probability that a record from this country is counterfeit
# (slight variation but no country is dramatically safer/riskier)
COUNTRY_CF_PROB = {
    "India":       0.23,
    "USA":         0.18,
    "UK":          0.20,
    "Germany":     0.19,
    "Switzerland": 0.21,
}

TODAY = datetime(2026, 7, 27)


def random_date(start: datetime, end: datetime) -> datetime:
    """Return a random datetime between start and end."""
    delta = (end - start).days
    if delta <= 0:
        return start
    return start + timedelta(days=random.randint(0, delta))


def generate_batch_number(is_counterfeit: bool) -> str:
    """
    Generate a batch number.

    Genuine:     'B' + 5-6 digits  (e.g. B12345, B789012)
    Counterfeit: Mix of:
      - Short random alphanumeric (2-4 chars) -- 40%
      - Normal-looking B+digits format         -- 35%
      - Random uppercase letters + digits      -- 25%
    """
    if not is_counterfeit:
        # Genuine: standard format, small chance of short batch
        if random.random() < 0.05:
            return "B" + str(random.randint(100, 9999))          # short genuine (noise)
        length = random.choice([5, 6])
        return "B" + str(random.randint(10**(length-1), 10**length - 1))

    else:
        r = random.random()
        if r < 0.40:
            # Short (2-4 chars) — suspicious
            chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            length = random.randint(2, 4)
            return "".join(random.choices(chars, k=length))
        elif r < 0.75:
            # Looks normal but actually counterfeit (overlap/noise)
            length = random.choice([5, 6])
            return "B" + str(random.randint(10**(length-1), 10**length - 1))
        else:
            # Mixed format
            prefix = random.choice(["CF", "MX", "RX", "GN"])
            return prefix + str(random.randint(100, 9999))


def pick_manufacturer(is_counterfeit: bool):
    """
    Pick a manufacturer.

    Genuine:     Mostly Tier 1 & 2, rarely Tier 3.
    Counterfeit: More likely Tier 2 & 3, but sometimes Tier 1 (overlap).
    """
    tier1 = [m for m, t in MANUFACTURERS if t == 1]
    tier2 = [m for m, t in MANUFACTURERS if t == 2]
    tier3 = [m for m, t in MANUFACTURERS if t == 3]

    if not is_counterfeit:
        weights = [0.60, 0.35, 0.05]          # mostly tier1, some tier2, rare tier3
    else:
        weights = [0.20, 0.40, 0.40]          # more tier2/tier3 for counterfeits

    chosen_tier = random.choices([tier1, tier2, tier3], weights=weights, k=1)[0]
    return random.choice(chosen_tier)


def generate_price(drug_baseline: int, is_counterfeit: bool) -> float:
    """
    Generate a price in INR.

    Genuine:     baseline ± 25%  (normal variation)
    Counterfeit: 25%-75% of baseline  (but 15% overlap at normal range)
    """
    if not is_counterfeit:
        # Genuine price: ±25% around baseline, log-normal noise
        price = drug_baseline * np.random.lognormal(mean=0.0, sigma=0.20)
        price = max(10.0, price)
    else:
        r = random.random()
        if r < 0.15:
            # Overlap: counterfeit priced like genuine (noise)
            price = drug_baseline * np.random.lognormal(mean=0.0, sigma=0.20)
        else:
            # Lower price: 20% - 70% of baseline
            ratio = random.uniform(0.20, 0.70)
            price = drug_baseline * ratio * np.random.lognormal(mean=0.0, sigma=0.15)
        price = max(5.0, price)

    return round(price, 2)


def generate_dates(is_counterfeit: bool):
    """
    Generate manufacture and expiry dates.

    Genuine:     manufactured 0-3 years ago, shelf life 1-3 years.
    Counterfeit: more likely to have very short shelf life (1-6 months),
                 but 20% have normal shelf life (overlap).
    """
    # Manufacture date: 0-3 years before today
    mfr_date = random_date(TODAY - timedelta(days=3*365), TODAY - timedelta(days=1))

    if not is_counterfeit:
        # Genuine: shelf life 365-1095 days (1-3 years)
        shelf_days = random.randint(365, 1095)
    else:
        r = random.random()
        if r < 0.20:
            # Overlap: normal shelf life
            shelf_days = random.randint(365, 1095)
        elif r < 0.65:
            # Short shelf life: 30-180 days (suspicious)
            shelf_days = random.randint(30, 180)
        else:
            # Medium: 180-365 days
            shelf_days = random.randint(180, 365)

    exp_date = mfr_date + timedelta(days=shelf_days)
    return mfr_date, exp_date


def assign_label_probabilistically(price_ratio: float,
                                   shelf_days: int,
                                   batch_len: int,
                                   mfr_tier: int,
                                   intended_cf: bool) -> int:
    """
    Assign label probabilistically based on the generated features.

    This adds realistic noise: even records intended as counterfeit
    can get label=1 (genuine) if their features look benign, and vice versa.

    This ensures the dataset is NOT a perfect reproduction of if-statements.
    The label is the ground truth, but with noise.

    Returns: 1 (genuine) or 0 (counterfeit)
    """
    # Base probability of being counterfeit for this record
    if intended_cf:
        base_p_cf = 0.80      # mostly labelled counterfeit, but not always
    else:
        base_p_cf = 0.05      # mostly labelled genuine, but small noise

    # Adjust based on actual feature values (slight nudge, not deterministic)
    # Price ratio very low -> slight nudge toward counterfeit
    if price_ratio < 0.50:
        base_p_cf = min(base_p_cf + 0.08, 1.0)
    elif price_ratio > 1.40:
        base_p_cf = max(base_p_cf - 0.05, 0.0)

    # Very short shelf life -> slight nudge toward counterfeit
    if shelf_days < 90:
        base_p_cf = min(base_p_cf + 0.06, 1.0)

    # Very short batch number -> slight nudge toward counterfeit
    if batch_len <= 3:
        base_p_cf = min(base_p_cf + 0.05, 1.0)

    # Rare manufacturer -> slight nudge toward counterfeit
    if mfr_tier == 3:
        base_p_cf = min(base_p_cf + 0.04, 1.0)

    label = 0 if random.random() < base_p_cf else 1
    return label


def generate_dataset(n_rows: int) -> pd.DataFrame:
    """Generate the full synthetic dataset."""

    rows = []
    n_counterfeit_target = int(n_rows * COUNTERFEIT_RATE)
    n_genuine_target     = n_rows - n_counterfeit_target

    # Decide intended class for each row
    intentions = ([True] * n_counterfeit_target) + ([False] * n_genuine_target)
    random.shuffle(intentions)

    drug_info = {name: (dosage, baseline) for name, dosage, baseline in DRUGS}
    mfr_tier  = {name: tier for name, tier in MANUFACTURERS}

    for i, intended_cf in enumerate(intentions, start=1):
        drug_name, dosage_mg, baseline = random.choice(DRUGS)
        country  = random.choices(
            list(COUNTRY_CF_PROB.keys()),
            weights=list(COUNTRY_CF_PROB.values()) if intended_cf
                    else [1 - v for v in COUNTRY_CF_PROB.values()],
            k=1
        )[0]

        manufacturer = pick_manufacturer(intended_cf)
        batch_number = generate_batch_number(intended_cf)
        price_inr    = generate_price(baseline, intended_cf)
        mfr_date, exp_date = generate_dates(intended_cf)

        shelf_days = (exp_date - mfr_date).days
        batch_len  = len(batch_number)
        price_ratio = price_inr / baseline   # rough ratio using population baseline

        label = assign_label_probabilistically(
            price_ratio, shelf_days, batch_len,
            mfr_tier.get(manufacturer, 2), intended_cf
        )

        rows.append({
            "drug_id"           : i,
            "drug_name"         : drug_name,
            "manufacturer"      : manufacturer,
            "batch_number"      : batch_number,
            "manufacture_country": country,
            "manufacture_date"  : mfr_date.strftime("%Y-%m-%d"),
            "expiry_date"       : exp_date.strftime("%Y-%m-%d"),
            "active_ingredient" : drug_name,
            "dosage_mg"         : dosage_mg,
            "price_inr"         : price_inr,
            "label"             : label,
        })

    df = pd.DataFrame(rows)
    return df


def main():
    print("Generating synthetic dataset...")
    df = generate_dataset(N_ROWS)

    # Basic sanity checks
    label_counts = df["label"].value_counts()
    print(f"  Total rows       : {len(df)}")
    print(f"  Genuine  (1)     : {label_counts.get(1, 0)}")
    print(f"  Counterfeit (0)  : {label_counts.get(0, 0)}")
    print(f"  Counterfeit rate : {label_counts.get(0, 0)/len(df)*100:.1f}%")
    print(f"  Columns          : {df.columns.tolist()}")
    print()

    df["manufacture_date"] = pd.to_datetime(df["manufacture_date"])
    df["expiry_date"]      = pd.to_datetime(df["expiry_date"])
    df["shelf_life"]       = (df["expiry_date"] - df["manufacture_date"]).dt.days
    df["batch_len"]        = df["batch_number"].str.len()

    print("  Shelf life (days) by label:")
    for lbl in [0, 1]:
        name = "Counterfeit" if lbl == 0 else "Genuine"
        grp  = df[df["label"] == lbl]["shelf_life"]
        print(f"    {name}: mean={grp.mean():.0f}, median={grp.median():.0f}, min={grp.min():.0f}, max={grp.max():.0f}")
    print()

    print("  Price (INR) by label:")
    for lbl in [0, 1]:
        name = "Counterfeit" if lbl == 0 else "Genuine"
        grp  = df[df["label"] == lbl]["price_inr"]
        print(f"    {name}: mean={grp.mean():.0f}, median={grp.median():.0f}")
    print()

    print("  Batch length by label:")
    for lbl in [0, 1]:
        name = "Counterfeit" if lbl == 0 else "Genuine"
        grp  = df[df["label"] == lbl]["batch_len"]
        print(f"    {name}: mean={grp.mean():.2f}, median={grp.median():.1f}")
    print()

    # Save (drop helper columns — they're recreated during feature engineering)
    df_save = df.drop(columns=["shelf_life", "batch_len"])
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_save.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
