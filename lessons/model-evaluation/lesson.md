# Model Evaluation and Generalization

## Overview

This lesson develops practical skills for the IOAI machine-learning syllabus. Examples are educational demonstrations, not official competition tasks or guarantees of real-world performance.

**Estimated time:** 3 hours  
**Difficulty:** intermediate

## Learning objectives

- Explain why training score alone is insufficient.
- Create train, validation, and test splits correctly.
- Calculate confusion-matrix metrics.
- Interpret precision, recall, F1, and ROC-AUC.
- Recognize leakage, class imbalance, and misleading metrics.

## 1. Generalization

A model generalizes when it performs well on examples drawn from the intended deployment setting. Training performance measures fit to training data, not necessarily performance on unseen examples. A model that memorizes noise may have excellent training performance but poor test performance.

```python
# Evaluate a model on data not used to fit it.
# Keep the test set isolated until final model selection is complete.
```

## 2. Data splitting

Training data fit model parameters. Validation data support model selection and threshold tuning. Test data provide a final estimate after choices are made. For small datasets, cross-validation can use data more efficiently, but preprocessing must be fitted separately within each training fold.

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
```

## 3. Confusion matrix

For binary classification, true positives (TP) are positive examples predicted positive; true negatives (TN) are negative examples predicted negative; false positives (FP) are negative examples predicted positive; false negatives (FN) are positive examples predicted negative.

```python
from sklearn.metrics import confusion_matrix

# y_true and y_pred contain binary labels
# tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
```

## 4. Precision, recall, and F1

Precision = TP/(TP+FP): among predicted positives, how many are truly positive? Recall = TP/(TP+FN): among actual positives, how many are detected? F1 is the harmonic mean of precision and recall. Handle zero denominators explicitly when implementing metrics.

```python
from sklearn.metrics import precision_score, recall_score, f1_score
print(precision_score(y_true, y_pred, zero_division=0))
print(recall_score(y_true, y_pred, zero_division=0))
print(f1_score(y_true, y_pred, zero_division=0))
```

## 5. Accuracy and imbalance

Accuracy is correct predictions divided by all predictions. It can hide poor minority-class performance. For example, predicting the majority class for every case can achieve high accuracy in a highly imbalanced dataset while detecting no minority examples.

```python
from sklearn.metrics import accuracy_score, balanced_accuracy_score
print(accuracy_score(y_true, y_pred))
print(balanced_accuracy_score(y_true, y_pred))
```

## 6. ROC curve and AUC

The ROC curve plots true-positive rate against false-positive rate across thresholds. ROC-AUC summarizes ranking ability across thresholds. It does not tell you whether a specific threshold meets operational needs; for highly imbalanced data, precision-recall analysis can be more informative.

```python
from sklearn.metrics import roc_auc_score, roc_curve

# Use probability scores for the positive class, not hard labels.
# auc = roc_auc_score(y_true, positive_probabilities)
# fpr, tpr, thresholds = roc_curve(y_true, positive_probabilities)
```

## 7. Cross-validation

In k-fold cross-validation, data are split into k folds; each fold serves as validation once while the others train the model. Report the distribution or mean and variation across folds. Use stratified folds for classification when appropriate. Grouped or time-aware splits are needed when samples are dependent.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# scores = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, cv=cv, scoring="f1")
```

## 8. Data leakage and honest evaluation

Leakage occurs when information unavailable at prediction time influences training or evaluation. Examples include scaling before splitting, using future data to predict the past, duplicate records crossing splits, and selecting features using test labels. Fit preprocessing within a pipeline and match the split strategy to how data will be used.

```python
# Correct pattern: put transformations and estimator in one sklearn Pipeline.
# Cross-validation then fits each transformation only on that fold's training partition.
```

## 9. Evaluation report

A useful report includes the split strategy, sample counts, class distribution, chosen metrics, baseline, uncertainty or variation when available, and limitations. Explain why the metrics match the task's costs rather than reporting one score without context.

## Mini-project

Complete this baseline implementation, run it, and explain the results. Change one design choice and describe how the output changes.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, proba))
```

## Common mistakes to avoid

- Do not assume that a successful run proves the result is correct.
- Check shapes, data types, missing values, and class distributions before modeling.
- Keep a final test set separate from model fitting and tuning.
- Use a reproducible random seed for teaching experiments and record the split strategy.
- Distinguish a numerical demonstration from evidence about real-world performance.

## IOAI practice checklist

- [ ] Explain the main idea without relying on code alone.
- [ ] Run and modify the examples.
- [ ] Test the approach on an edge case.
- [ ] Interpret the output in context.
- [ ] Record limitations and possible sources of error.

## Recommended resources

- IOAI Education Hub: https://ioai-official.org/resources/
- Kaggle Learn: https://www.kaggle.com/learn
- NumPy documentation: https://numpy.org/doc/stable/
- pandas documentation: https://pandas.pydata.org/docs/
- scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html
- Google Machine Learning Crash Course: https://developers.google.com/machine-learning/crash-course/
