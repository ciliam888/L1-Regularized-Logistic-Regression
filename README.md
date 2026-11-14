# IDS 570 – Homework 3: L1 vs. L2 Logistic Regression

Comparison of L2 (default) and L1 (lasso) regularized logistic regression
for the Week 09 merchant-discourse classifier (TF–IDF features).

## Files
- `step7_label_chunks.py` – weak-supervision labeling (Week 08 Step 7)
- `step9_prepare_datasets.py` – CORE vs. NEG downsampling + 80/20 split (Week 08 Step 9)
- `hw03_l2_baseline.py` – baseline model, L2 penalty
- `hw03_l1_model.py` – L1 model (`l1_ratio=1`, `solver="liblinear"`)
- `hw03_compare_errors.py` – chunks where L1 and L2 disagree (label-noise analysis)
- `hw03_l2_output.txt`, `hw03_l1_output.txt`, `hw03_compare_output.txt` – printed results

## Setup
Python 3.12, scikit-learn 1.9.1. Run in order:
1. `python step7_label_chunks.py`
2. `python step9_prepare_datasets.py`
3. `python hw03_l2_baseline.py`
4. `python hw03_l1_model.py`
5. `python hw03_compare_errors.py`

## Corpus note
571 .txt files as downloaded from Canvas ("Train_Text_Documents"), including
6 files with (1) whose content differs from the same-ID originals;
all were kept. Result: 9,773 CORE / 524,688 NEG / 469 MAYBE chunks;
train = 15,636, test = 3,910. The corpus and data files are not included.
