# Classification: Predicting Categories

## Overview

This lesson develops practical skills for the IOAI machine-learning syllabus. Examples are educational demonstrations, not official competition tasks or guarantees of real-world performance.

**Estimated time:** 3 hours  
**Difficulty:** intermediate

## Learning objectives

- Distinguish classification from regression.
- Understand labels, class probabilities, and decision thresholds.
- Train logistic regression, K-NN, and decision-tree classifiers.
- Recognize scaling needs and decision-boundary behavior.
- Use a reproducible train/test workflow.

## 1. Classification task

Classification predicts a discrete label such as spam/not spam or one of several image categories. Binary classification has two classes; multiclass classification has more than two. A label is not the same thing as a numerical measurement, even if it is encoded as 0 and 1.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
print(X.shape, y.shape)
print(data.target_names)
```

## 2. Train/test split

A held-out test set estimates performance on unseen examples. For classification, stratification can preserve approximate class proportions in each split. Use a fixed random seed when you need reproducible teaching examples.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## 3. Logistic regression

Despite its name, logistic regression is a classification algorithm. It maps a linear score through a logistic function to estimate a probability for binary classification. A threshold, often 0.5 by default, converts probabilities into predicted labels.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
model.fit(X_train, y_train)
labels = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## 4. K-nearest neighbors

K-NN predicts using nearby training examples. It is intuitive but can be slow at prediction time and sensitive to feature scales. Standardization is often important because distance calculations can otherwise be dominated by features with large numeric ranges.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
```

## 5. Decision trees

Decision trees split data using feature-based rules. They can model nonlinear relationships and are easy to inspect when small, but deep trees may overfit. Control complexity with parameters such as max_depth and min_samples_leaf.

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
print(tree.score(X_test, y_test))
```

## 6. Probabilities and thresholds

A probability estimate expresses model confidence, not guaranteed correctness. Lowering the positive-class threshold usually increases the number predicted positive; this often raises recall and may lower precision. Choose thresholds based on the problem's error costs using validation data.

```python
positive_probability = model.predict_proba(X_test)[:, 1]
threshold = 0.4
custom_labels = (positive_probability >= threshold).astype(int)
```

## 7. Multiclass and class imbalance

Multiclass tasks predict among three or more classes. In imbalanced data, accuracy can be high even when the minority class is rarely detected. Inspect class counts and metrics such as per-class recall, precision, and macro F1.

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, labels, target_names=data.target_names))
```

## 8. Mini-project

Train at least two classifiers using the same split. Compare their predictions and evaluation metrics. State which mistakes matter most in the application and do not choose a model based on test performance alone.

## Mini-project

Complete this baseline implementation, run it, and explain the results. Change one design choice and describe how the output changes.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred, target_names=data.target_names))
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
