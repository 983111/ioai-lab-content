# Statistics for Machine Learning

## Overview

This lesson develops practical skills for the IOAI machine-learning syllabus. Examples are educational demonstrations, not official competition tasks or guarantees of real-world performance.

**Estimated time:** 2–3 hours  
**Difficulty:** beginner

## Learning objectives

- Calculate and interpret mean, median, mode, range, variance, and standard deviation.
- Understand distributions, percentiles, and outliers.
- Distinguish correlation from causation.
- Use probability and sampling language accurately.
- Explain why summary statistics do not tell the whole story.

## 1. Population, sample, and variables

A population is the complete group of interest. A sample is the subset observed. A statistic is calculated from a sample; a population parameter describes the population. Numerical variables measure quantities, while categorical variables describe groups or labels.

```python
scores = [62, 70, 70, 82, 96]
print(len(scores))
print(sum(scores) / len(scores))
```

## 2. Mean, median, and mode

The arithmetic mean is the sum divided by the count. The median is the middle sorted value (or the average of two middle values). The mode is the most frequent value. The mean is sensitive to extreme values; the median is more robust.

```python
import statistics as st
values = [10, 11, 12, 12, 100]
print(st.mean(values))
print(st.median(values))
print(st.multimode(values))
```

## 3. Spread: range, variance, and standard deviation

Range is maximum minus minimum. Variance describes average squared deviation from a mean; standard deviation is its square root and is expressed in the original units. Sample variance uses n−1 in the denominator; population variance uses n.

```python
values = [2, 4, 4, 6]
print(max(values) - min(values))
print(st.pvariance(values))  # population variance
print(st.pstdev(values))     # population standard deviation
print(st.variance(values))    # sample variance
```

## 4. Percentiles and distributions

A percentile describes a position in a distribution. The 50th percentile is the median under common definitions, though interpolation conventions can differ. A histogram helps reveal skew, multiple peaks, and unusual values.

```python
import numpy as np
values = np.array([12, 15, 17, 18, 20, 22, 30])
print(np.percentile(values, [25, 50, 75]))
```

## 5. Probability and sampling

Probability ranges from 0 to 1. For equally likely outcomes, probability is favorable outcomes divided by all outcomes. Sample statistics vary from sample to sample; larger representative samples often reduce sampling variability, but size alone does not fix selection bias.

```python
# Equally likely outcomes for a fair six-sided die
probability_even = 3 / 6
print(probability_even)

# Simulate repeated fair-die rolls
rng = np.random.default_rng(7)
rolls = rng.integers(1, 7, size=10000)
print(np.mean(rolls % 2 == 0))
```

## 6. Correlation is not causation

Correlation measures association. Pearson correlation captures linear association and ranges from -1 to 1. Correlation does not establish that one variable causes another; confounding, reverse causality, and coincidence may explain a relationship.

```python
hours = np.array([1, 2, 3, 4, 5])
scores = np.array([50, 60, 65, 75, 80])
print(np.corrcoef(hours, scores)[0, 1])
```

## 7. Outliers and robust summaries

Outliers may be errors, rare valid cases, or important events. Investigate units and context before removing them. The interquartile range (IQR) is Q3 − Q1; a common exploratory rule flags values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR.

```python
q1, q3 = np.percentile(values, [25, 75])
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print(lower, upper)
```

## 8. Why statistics matters in AI

Class imbalance, noisy measurements, skewed features, and small samples can affect model training and evaluation. Inspect distributions and sample sizes before interpreting a metric. A single average may hide important subgroups.

## Mini-project

Complete this baseline implementation, run it, and explain the results. Change one design choice and describe how the output changes.

```python
import numpy as np

def describe_values(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("values must not be empty")
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "std_population": float(np.std(values))
    }
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
