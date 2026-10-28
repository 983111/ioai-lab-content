# Data Manipulation with NumPy and pandas

## Overview

This lesson develops practical skills for the IOAI machine-learning syllabus. Examples are educational demonstrations, not official competition tasks or guarantees of real-world performance.

**Estimated time:** 2–3 hours  
**Difficulty:** beginner

## Learning objectives

- Represent numerical data with NumPy arrays.
- Inspect and transform tabular data with pandas.
- Filter rows and select columns safely.
- Handle missing values and categorical columns.
- Create simple plots and avoid data leakage during preparation.

## 1. NumPy arrays

NumPy provides efficient numerical arrays and vectorized operations. Unlike Python lists, arrays are designed for numerical computation and typically operate element-wise when their shapes are compatible.

```python
import numpy as np

values = np.array([10, 20, 30, 40])
print(values.mean())       # 25.0
print(values * 2)          # [20 40 60 80]
print(values.shape)        # (4,)
print(values.dtype)
```

## 2. Dimensions, indexing, and slicing

A one-dimensional array is a vector; a two-dimensional array can represent rows and columns. Shape helps you understand the layout of the data.

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.shape)   # (2, 3)
print(matrix[0, 1])   # 2
print(matrix[:, 0])   # first column
print(matrix[1, :])   # second row
```

## 3. pandas DataFrames

A DataFrame is a labeled table. Each column can have its own data type, and column names make operations easier to read.

```python
import pandas as pd

df = pd.DataFrame({
    "hours": [2, 4, 6, 8],
    "score": [55, 68, 80, 92],
    "group": ["A", "A", "B", "B"]
})
print(df.head())
print(df.describe(numeric_only=True))
print(df["score"])
```

## 4. Selecting and filtering

Use column selection to choose variables and Boolean conditions to choose rows. Combine conditions with parentheses and the operators & and | for pandas Series.

```python
high_score = df[df["score"] >= 75]
selected = df.loc[df["hours"] >= 4, ["hours", "score"]]
print(high_score)
print(selected)
```

## 5. Missing values and data types

Missing values can affect summaries and model training. Inspect missingness before deciding whether to remove rows, impute values, or preserve a missingness indicator. The right choice depends on why values are missing.

```python
df.loc[1, "score"] = np.nan
print(df.isna().sum())
df["score_filled"] = df["score"].fillna(df["score"].median())
print(df["score_filled"])

# Convert only when the source values can be parsed:
# df["hours"] = pd.to_numeric(df["hours"], errors="coerce")
```

## 6. Encoding and scaling

Many models need numerical inputs. One-hot encoding represents categories using indicator columns. Scaling changes numerical ranges; fit scaling parameters on training data only, then apply the same transformation to validation and test data.

```python
encoded = pd.get_dummies(df, columns=["group"], dtype=int)
print(encoded.head())

# In a model workflow, use sklearn.preprocessing.StandardScaler
# inside a Pipeline so it is fitted only on the training folds.
```

## 7. Visualization and data quality

Visualizations can reveal distributions, outliers, and relationships. A plot does not prove causation. Check duplicates, units, impossible values, and whether columns mean what their names suggest.

```python
import matplotlib.pyplot as plt

plt.scatter([2, 4, 6, 8], [55, 68, 80, 92])
plt.xlabel("Study hours")
plt.ylabel("Score")
plt.title("Study hours and score (illustrative data)")
plt.show()
```

## 8. Mini-project: prepare a small table

Build a reproducible preparation routine: inspect shape and types, check missing values, select useful columns, handle missing data according to a stated rule, and document transformations. Never calculate preprocessing statistics from the test set.

## Mini-project

Complete this baseline implementation, run it, and explain the results. Change one design choice and describe how the output changes.

```python
import pandas as pd

def prepare_table(df):
    """Return a copy with rows missing the target removed and numeric missing values median-imputed."""
    result = df.copy()
    if "target" not in result.columns:
        raise ValueError("target column is required")
    result = result.dropna(subset=["target"])
    numeric_columns = result.select_dtypes(include="number").columns.drop("target", errors="ignore")
    for column in numeric_columns:
        result[column] = result[column].fillna(result[column].median())
    return result
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
