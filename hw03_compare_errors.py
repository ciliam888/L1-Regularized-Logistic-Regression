"""
HW03 - Compare L1 vs L2 errors to judge robustness to weak-label noise.
Shows test chunks where the two models disagree, plus the trigger word(s)
that produced each chunk's weak label.
"""
from pathlib import Path
import json
import numpy as np
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_DIR = Path("data")
TRIGGERS = {"merchant", "merchants", "marchant", "marchants",
            "factor", "chapman", "adventurer", "adventurers",
            "venturer", "venturers", "staple", "staplers", "trade", "purser"}

with open(DATA_DIR / "train_core_vs_neg.json", encoding="utf-8") as f:
    train = json.load(f)
with open(DATA_DIR / "test_core_vs_neg.json", encoding="utf-8") as f:
    test = json.load(f)

Xtr, ytr = [t for t, y in train], [y for t, y in train]
Xte, yte = [t for t, y in test], np.array([y for t, y in test])

vec = TfidfVectorizer(lowercase=True, min_df=5, max_df=0.9)
A, B = vec.fit_transform(Xtr), vec.transform(Xte)

l2 = LogisticRegression(max_iter=1000).fit(A, ytr)
l1 = LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=2000).fit(A, ytr)

p2, p1 = l2.predict(B), l1.predict(B)
pr2, pr1 = l2.predict_proba(B)[:, 1], l1.predict_proba(B)[:, 1]

def triggers_in(text):
    return sorted(set(simple_preprocess(text, deacc=True)) & TRIGGERS)

# L2 wrong, L1 agrees with the label -> where label noise would show up
l2_fn = np.where((yte == 1) & (p2 == 0) & (p1 == 1))[0]   # label CORE, L2 says NEG
l2_fp = np.where((yte == 0) & (p2 == 1) & (p1 == 0))[0]   # label NEG,  L2 says CORE
both_wrong = np.where((p2 != yte) & (p1 != yte))[0]

print(f"Test chunks where L1 and L2 disagree: {(p1 != p2).sum()}")
print(f"  L2 says NEG, label & L1 say CORE: {len(l2_fn)}")
print(f"  L2 says CORE, label & L1 say NEG: {len(l2_fp)}")
print(f"  Both models wrong:                {len(both_wrong)}")

def show(idx, title, n=10):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)
    for i in idx[:n]:
        text = " ".join(Xte[i].split())
        print(f"\n[test #{i}] label={yte[i]} | P_L2={pr2[i]:.3f} | P_L1={pr1[i]:.3f}"
              f" | triggers={triggers_in(Xte[i]) or 'none'}")
        print(f"  {text[:500]}")
        print("  YOUR VERDICT (merchant discourse? label right or model right?): ____")

show(l2_fn, "Label = CORE, L2 = NEG, L1 = CORE  (is the label really merchant discourse?)")
show(l2_fp, "Label = NEG, L2 = CORE, L1 = NEG  (did L2 find merchant discourse the rule missed?)")
show(both_wrong, "Both models disagree with the label", n=8)