"""
HW03 - Baseline: TF-IDF + Logistic Regression with L2 (default) regularization.
Reproduces the Week 09 tutorial classifier and adds interpretability diagnostics.
"""
from pathlib import Path
import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

DATA_DIR = Path("data")
MODEL_NAME = "L2"

# 1. Load the saved train/test splits (do NOT re-split: keeps the comparison fair)
with open(DATA_DIR / "train_core_vs_neg.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open(DATA_DIR / "test_core_vs_neg.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

X_train_texts = [t for (t, y) in train_data]
y_train = [y for (t, y) in train_data]
X_test_texts = [t for (t, y) in test_data]
y_test = [y for (t, y) in test_data]

print(f"Train size: {len(X_train_texts)} | Test size: {len(X_test_texts)}")

# 2. TF-IDF: same settings as the tutorial
vectorizer = TfidfVectorizer(lowercase=True, min_df=5, max_df=0.9)
X_train = vectorizer.fit_transform(X_train_texts)   # fit on train only
X_test = vectorizer.transform(X_test_texts)
print("TF-IDF shape (train):", X_train.shape)

# 3. Train logistic regression (scikit-learn default = L2 penalty, C=1.0)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# 4. Evaluation
print(f"\n===== {MODEL_NAME} MODEL: EVALUATION =====")
print("\nConfusion matrix (rows = actual, cols = predicted):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))

print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))

# 5. Sparsity diagnostic
coefs = clf.coef_[0]
n_total = len(coefs)
n_nonzero = np.count_nonzero(coefs)
print(f"\nNon-zero coefficients: {n_nonzero} / {n_total} "
      f"({n_nonzero / n_total:.1%})")

# 6. What the model learned: top positive / negative words
feature_names = vectorizer.get_feature_names_out()
order = np.argsort(coefs)            # ascending: most negative first

print(f"\nTop 15 POSITIVE words (predict CORE = 1):")
for i in order[::-1][:15]:
    if coefs[i] > 0:
        print(f"  {feature_names[i]:20s} {coefs[i]:+.4f}")

print(f"\nTop 15 NEGATIVE words (predict NEG = 0):")
neg_idx = [i for i in order if coefs[i] < 0][:15]
for i in neg_idx:
    print(f"  {feature_names[i]:20s} {coefs[i]:+.4f}")
if len(neg_idx) < 15:
    print(f"  (only {len(neg_idx)} coefficients are strictly negative)")

# 7. Misclassified examples (for the label-noise question)
y_test_arr = np.array(y_test)
fp_idx = np.where((y_test_arr == 0) & (y_pred == 1))[0]   # labeled NEG, predicted CORE
fn_idx = np.where((y_test_arr == 1) & (y_pred == 0))[0]   # labeled CORE, predicted NEG

print(f"\nFalse positives: {len(fp_idx)} | False negatives: {len(fn_idx)}")

def show(indices, kind, n=5):
    print(f"\n--- Sample {kind} ---")
    for i in indices[:n]:
        text = " ".join(X_test_texts[i].split())
        print(f"\n[test #{i}] P(CORE)={y_prob[i]:.3f}")
        print(f"  {text[:400]}...")

show(fp_idx, "FALSE POSITIVES (label NEG, model says CORE)")
show(fn_idx, "FALSE NEGATIVES (label CORE, model says NEG)")

# Save all misclassifications so we can compare L1 vs L2 later
misclassified = [
    {"test_index": int(i), "true_label": int(y_test[i]), "pred_label": int(y_pred[i]),
     "prob_core": float(y_prob[i]), "text": X_test_texts[i]}
    for i in np.concatenate([fp_idx, fn_idx])
]
out_path = DATA_DIR / f"hw03_{MODEL_NAME.lower()}_misclassified.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(misclassified, f, ensure_ascii=False, indent=2)
print(f"\nSaved {len(misclassified)} misclassified chunks to {out_path}")