import os, sys, pickle, datetime, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'counterfeit_model.pkl')
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'medicine_dataset.csv')

with open(MODEL_PATH, 'rb') as f:
    saved = pickle.load(f)
model = saved['model']
stats = saved['stats']

from feature_engineering import create_single_features, FEATURE_COLS, create_features

# ── Q4: Engineered features for suspicious input ──
print()
print("Q4 - ENGINEERED FEATURES for Suspicious Input")
print("  Input: Paracetamol, Sun Pharma, Batch=A1, India,")
print("         Manufacture=2026-07-01, Expiry=2026-08-01, Dosage=500, Price=10")
print()

form_suspicious = {
    "medicine_name"   : "Paracetamol",
    "manufacturer"    : "Sun Pharma",
    "batch_number"    : "A1",
    "country"         : "India",
    "manufacture_date": "2026-07-01",
    "expiry_date"     : "2026-08-01",
    "dosage"          : 500,
    "price"           : 10,
}

feat_susp = create_single_features(form_suspicious, stats)
feature_names = ["price_ratio","days_until_expiry","medicine_age",
                 "shelf_life","batch_length","manufacturer_frequency",
                 "country_risk","price_per_dose"]

for name, val in zip(feature_names, feat_susp[0]):
    print(f"  {name:<30} = {val:.4f}")

# ── Q11: Currency check ──
print()
print("Q11 - PRICE / CURRENCY CONSISTENCY")
avg_prices = stats.get("average_price") or stats.get("drug_avg_price", {})
para_avg = avg_prices.get("Paracetamol", 0)
print(f"  Dataset column          : price_usd (USD, range $1-$150, mean ~$75)")
print(f"  Dataset Paracetamol avg : ${para_avg:.2f} USD")
print(f"  Flask UI shows          : Rs (Indian Rupees)")
print(f"  User entered            : 10  (meaning Rs 10 INR)")
print(f"  price_ratio computed as : 10 / {para_avg:.2f} = {10/para_avg:.4f}")
print(f"  PROBLEM: Rs10 INR treated as $10 USD. $10 / $75 = 0.13. Price looks low but")
print(f"  this feature has near-zero predictive value in training data (prices are random).")

# ── Q3: model.predict + model.predict_proba ──
print()
print("Q3 - model.predict and model.predict_proba for Suspicious Input")
print()

prediction_susp = model.predict(feat_susp)[0]
probas_susp     = model.predict_proba(feat_susp)[0]

classes = model.classes_
counterfeit_idx = list(classes).index(0)
genuine_idx     = list(classes).index(1)
p_cf = probas_susp[counterfeit_idx]
p_gn = probas_susp[genuine_idx]

print(f"  model.predict(features)         = {prediction_susp}")
print(f"  model.predict_proba(features)   = {probas_susp}")
print(f"  model.classes_                  = {classes}")
print(f"  P(Counterfeit) [index {counterfeit_idx}]         = {p_cf:.6f}  ({p_cf*100:.2f}%)")
print(f"  P(Genuine)     [index {genuine_idx}]         = {p_gn:.6f}  ({p_gn*100:.2f}%)")
print(f"  Predicted class                 = {prediction_susp}  (GENUINE)")

# ── Q6: How predict.py maps probas to risk ──
print()
print("Q6 - HOW predict.py CONVERTS probabilities TO RISK")
print()
print("  CODE in predict.py (lines 72-85):")
print("    prediction    = model.predict(features)[0]")
print("    probabilities = model.predict_proba(features)[0]")
print("    confidence    = round(max(probabilities) * 100, 1)    # <- PROBLEM HERE")
print("    if prediction == 1:      risk = 'Low'")
print("    elif confidence >= 70:   risk = 'High'")
print("    else:                    risk = 'Medium'")
print()
conf = round(max(probas_susp) * 100, 1)
risk = "Low" if prediction_susp == 1 else ("High" if conf >= 70 else "Medium")
print(f"  Applied to suspicious input:")
print(f"    max(probabilities)   = {max(probas_susp):.6f}")
print(f"    confidence           = {conf}%")
print(f"    prediction == 1?     = {prediction_susp == 1}  ->  risk = '{risk}'")
print()
print("  PROBLEM 1: confidence = max(proba). When prediction=1 (Genuine), the max is")
print("  P(Genuine). So 99.9% confidence = 99.9% chance of being Genuine. Displayed as")
print("  'Low Risk 99.9%' which is technically correct -- but misleading.")
print("  PROBLEM 2: model.classes_ is not used to extract counterfeit probability.")

# ── Q8: Was model retrained? ──
print()
print("Q8 - WAS MODEL RETRAINED?")
pkl_mtime  = datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
fe_mtime   = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(BASE_DIR,'feature_engineering.py')))
trn_mtime  = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(BASE_DIR,'train_model.py')))
pred_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(BASE_DIR,'predict.py')))
print(f"  counterfeit_model.pkl    last modified: {pkl_mtime}")
print(f"  feature_engineering.py   last modified: {fe_mtime}")
print(f"  train_model.py           last modified: {trn_mtime}")
print(f"  predict.py               last modified: {pred_mtime}")
print()
print("  The PKL was created on 2026-07-24. NO changes were made to any code files.")
print("  The plan was presented but NOT executed. The old model is still running.")

# ── Q9: Model evaluation on test set ──
print()
print("Q9 - MODEL EVALUATION ON CURRENT TEST SET")
df = pd.read_csv(DATA_PATH)
df["manufacturer"]     = df["manufacturer"].astype(str).str.strip().str.strip(',"')
df["manufacture_date"] = pd.to_datetime(df["manufacture_date"], errors="coerce")
df["expiry_date"]      = pd.to_datetime(df["expiry_date"],      errors="coerce")
df = df.dropna(subset=["manufacture_date","expiry_date"])
df = df[df["expiry_date"] > df["manufacture_date"]]
df = df[df["price_usd"] > 0]
df = df.dropna(subset=["label"])

df, _ = create_features(df)
X = df[FEATURE_COLS].fillna(0)
y = df["label"]

_, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
y_pred = model.predict(X_test)

print(f"  Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["Counterfeit(0)", "Genuine(1)"]))
cm = confusion_matrix(y_test, y_pred)
print("  Confusion Matrix (rows=actual, cols=predicted):")
print(f"                     Pred Counterfeit   Pred Genuine")
print(f"  Actual Counterfeit       {cm[0,0]:6d}         {cm[0,1]:6d}")
print(f"  Actual Genuine           {cm[1,0]:6d}         {cm[1,1]:6d}")
print()
print(f"  NOTE: {cm[0,1]} counterfeit medicines are MISCLASSIFIED as Genuine (False Negatives).")
print(f"  This is medically dangerous. Recall for Counterfeit = {cm[0,0]/(cm[0,0]+cm[0,1]):.2%}")

# ── Q10: Feature value comparison ──
print()
print("Q10 - FEATURE COMPARISON: Suspicious Input vs Training Distributions")
print()
genuine_rows     = df[df["label"] == 1][FEATURE_COLS]
counterfeit_rows = df[df["label"] == 0][FEATURE_COLS]

hdr = f"  {'Feature':<26} {'Input':>8} {'Gen.Mean':>10} {'Gen.Med':>8} {'CF.Mean':>9} {'CF.Med':>8}  Note"
print(hdr)
print("  " + "-"*90)
notes = {
    "price_ratio"           : "LOW but not informative - same dist in training",
    "days_until_expiry"     : "Only 4 days - near expired",
    "medicine_age"          : "26 days old - very new",
    "shelf_life"            : "31 days - extremely short",
    "batch_length"          : "2 chars - ALL training batches are 7 chars",
    "manufacturer_frequency": "4731 - Sun Pharma is very common -> GENUINE signal",
    "country_risk"          : "~0.15 - same for all countries, no signal",
    "price_per_dose"        : "0.02 - very low but not informative in training",
}
for i, name in enumerate(feature_names):
    sv  = feat_susp[0][i]
    gm  = genuine_rows[name].mean()
    gmd = genuine_rows[name].median()
    cfm = counterfeit_rows[name].mean()
    cfmd= counterfeit_rows[name].median()
    note= notes.get(name,"")
    print(f"  {name:<26} {sv:>8.3f} {gm:>10.3f} {gmd:>8.3f} {cfm:>9.3f} {cfmd:>8.3f}  {note}")

# ── Q12: Data leakage ──
print()
print("Q12 - DATA LEAKAGE")
print()
print("  create_features() in feature_engineering.py computes:")
print("    country_risk = df.groupby('manufacture_country')['label'].apply(lambda v: 1-v.mean())")
print("    This uses the FULL df (all 50,000 rows) BEFORE train_test_split.")
print("    -> country_risk is calculated using test-set labels. LEAKAGE EXISTS.")
print()
print("    manufacturer_frequency = df['manufacturer'].value_counts()")
print("    Also uses full df. Minor leakage.")
print()
print("  Impact: Both features have near-zero class separation, so leakage doesn't")
print("  artificially inflate accuracy in a detectable way -- but it's still wrong.")

# ── Q13: Reasons origin ──
print()
print("Q13 - ORIGIN OF 'REASONS' DISPLAYED TO USER")
print()
print("  generate_reasons() in predict.py uses ONLY hardcoded rule thresholds.")
print("  It does NOT use model.predict_proba(), model.coef_, or any model output.")
print("  It runs independently. The ML model is completely ignored by this function.")
print()
print("  Specific rules:")
print("    1. price < 0.70 x avg_price       -> 'Price is X% lower than average'")
print("    2. manufacturer_count == 0         -> 'Manufacturer not found'")
print("    3. manufacturer_count < 20         -> 'Very few records'")
print("    4. country_risk > 0.25             -> 'Higher-risk country'")
print("    5. len(batch) < 4                  -> 'Batch unusually short'")
print()
print("  ANSWER: The 'Reasons' are RULE-BASED, NOT model-derived.")
print("  The UI section should be renamed to 'Observed Risk Factors' to be honest.")

# ── Logistic Regression coefficients ──
print()
print("BONUS - LOGISTIC REGRESSION COEFFICIENTS")
print("  (what drives prediction: positive = pushes toward Genuine, negative = Counterfeit)")
print()
coef_pairs = list(zip(feature_names, model.coef_[0]))
for name, coef in sorted(coef_pairs, key=lambda x: -abs(x[1])):
    direction = "-> Genuine" if coef > 0 else "-> Counterfeit"
    print(f"  {name:<30} coef={coef:+.4f}  {direction}")

print()
print("  KEY FINDING: batch_length has coef=-0.9243 (pushes toward Counterfeit)")
print("  But batch=A1 gives batch_length=2.  Logit contribution = -0.9243 * 2 = -1.85")
print("  manufacturer_frequency coef=+0.0020. Sun Pharma freq=4731.")
print("  Logit contribution = +0.0020 * 4731 = +9.46")
print("  Net logit = +9.46 - 1.85 + other small terms = strongly Genuine -> P(Genuine)~99.9%")
print()
print("  CONCLUSION: The massive manufacturer_frequency value (4731) completely overwhelms")
print("  the batch_length signal (-1.85 vs +9.46). The model was never taught to distrust")
print("  common manufacturers for suspicious batch inputs.")
