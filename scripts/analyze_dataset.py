"""
Dataset analysis script to understand the data and identify problems.
"""

import pandas as pd
import numpy as np

df = pd.read_csv('data/medicine_dataset.csv')
df['manufacture_date'] = pd.to_datetime(df['manufacture_date'], errors='coerce')
df['expiry_date'] = pd.to_datetime(df['expiry_date'], errors='coerce')

print("=== WHAT DISTINGUISHES COUNTERFEIT FROM GENUINE? ===\n")

# Price ratio within drug group
avg_price = df.groupby('drug_name')['price_usd'].mean()
df['price_ratio'] = df['price_usd'] / df['drug_name'].map(avg_price)
for label in [0, 1]:
    name = 'Counterfeit' if label == 0 else 'Genuine'
    grp = df[df['label'] == label]['price_ratio']
    print(f'Price Ratio {name}: mean={grp.mean():.4f}, std={grp.std():.4f}')
print()

# Manufacturer name leak
df['has_fake_in_name'] = df['manufacturer'].str.contains('Fake', case=False, na=False)
print("Has 'Fake_' in manufacturer name vs label:")
print(pd.crosstab(df['has_fake_in_name'], df['label']))
print()

fake_mfr = df[df['has_fake_in_name']]
real_mfr = df[~df['has_fake_in_name']]
fake_label0_rate = (fake_mfr['label'] == 0).mean()
real_label0_rate = (real_mfr['label'] == 0).mean()
print(f"Fake manufacturer rows: {len(fake_mfr)}, label=0 rate: {fake_label0_rate:.2%}")
print(f"Real manufacturer rows: {len(real_mfr)}, label=0 rate: {real_label0_rate:.2%}")
print()

# How many label=0 without Fake_ in name?
suspicious = df[(df['label'] == 0) & (~df['has_fake_in_name'])]
print(f"Label=0 WITHOUT 'Fake_' in manufacturer: {len(suspicious)}")
if len(suspicious) > 0:
    print("  Their manufacturers (top 10):")
    print(suspicious['manufacturer'].value_counts().head(10))
print()

# Drug-level price comparison
print("=== DRUG-LEVEL PRICE COMPARISON (genuine vs counterfeit) ===")
for drug in df['drug_name'].unique()[:8]:
    sub = df[df['drug_name'] == drug]
    g_price = sub[sub['label'] == 1]['price_usd'].mean()
    cf_count = (sub['label'] == 0).sum()
    c_price = sub[sub['label'] == 0]['price_usd'].mean() if cf_count > 0 else float('nan')
    print(f"  {drug}: Genuine avg={g_price:.2f}, Counterfeit avg={c_price:.2f} (n={cf_count})")
print()

print("=== ALL DRUG NAMES ===")
print(df['drug_name'].value_counts())
print()

# Manufacturer frequency
df['mfr_freq'] = df['manufacturer'].map(df['manufacturer'].value_counts())
print("=== MANUFACTURER FREQUENCY BY LABEL ===")
for label in [0, 1]:
    name = 'Counterfeit' if label == 0 else 'Genuine'
    grp = df[df['label'] == label]['mfr_freq']
    print(f"Mfr Frequency {name}: mean={grp.mean():.1f}, median={grp.median():.1f}")
print()

# Country risk (using all data - data leakage!)
country_risk = df.groupby('manufacture_country')['label'].apply(lambda v: 1 - v.mean()).to_dict()
df['country_risk'] = df['manufacture_country'].map(country_risk)
print("=== COUNTRY RISK (from full dataset) ===")
for label in [0, 1]:
    name = 'Counterfeit' if label == 0 else 'Genuine'
    grp = df[df['label'] == label]['country_risk']
    print(f"Country Risk {name}: mean={grp.mean():.4f}, std={grp.std():.4f}")
print()

# Shelf life
df['shelf_life_days'] = (df['expiry_date'] - df['manufacture_date']).dt.days
print("=== SHELF LIFE BY LABEL ===")
for label in [0, 1]:
    name = 'Counterfeit' if label == 0 else 'Genuine'
    grp = df[df['label'] == label]['shelf_life_days']
    print(f"Shelf Life {name}: mean={grp.mean():.1f}, std={grp.std():.1f}")
print()

# Batch length
df['batch_len'] = df['batch_number'].astype(str).str.len()
print("=== BATCH LENGTH BY LABEL ===")
for label in [0, 1]:
    name = 'Counterfeit' if label == 0 else 'Genuine'
    grp = df[df['label'] == label]['batch_len']
    print(f"Batch Length {name}: mean={grp.mean():.2f}, std={grp.std():.2f}")
print()

# Sample batch numbers
print("=== SAMPLE BATCH NUMBERS ===")
print("Counterfeit samples:")
print(df[df['label'] == 0]['batch_number'].head(10).tolist())
print("Genuine samples:")
print(df[df['label'] == 1]['batch_number'].head(10).tolist())
print()

# How is the label actually assigned in this synthetic dataset?
# Check if any non-price feature actually correlates with label
from scipy.stats import pointbiserialr
import warnings
warnings.filterwarnings('ignore')

print("=== CORRELATIONS WITH LABEL ===")
numeric_cols = ['price_usd', 'dosage_mg', 'price_ratio', 'shelf_life_days', 'batch_len', 'mfr_freq']
for col in numeric_cols:
    corr, pval = pointbiserialr(df['label'], df[col])
    print(f"  {col}: r={corr:.4f}, p={pval:.4e}")
print()

print("=== CONCLUSION: WHAT MAKES A RECORD COUNTERFEIT? ===")
print("If label==0 rows are ONLY those with Fake_ in manufacturer name,")
print("then manufacturer_name is a trivially identifying feature NOT available at test time.")
print()
print(f"All label=0 have Fake_ in name: {df[(df['label']==0) & (~df['has_fake_in_name'])].empty}")
print(f"All label=1 do NOT have Fake_ in name: {df[(df['label']==1) & (df['has_fake_in_name'])].empty}")
