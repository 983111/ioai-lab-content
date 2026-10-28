# Regression: Predicting Numerical Values

## Overview

This lesson develops practical skills for the IOAI machine-learning syllabus. Examples are educational demonstrations, not official competition tasks or guarantees of real-world performance.

**Estimated time:** 3 hours  
**Difficulty:** intermediate

## Learning objectives

- Distinguish features from a numerical target.
- Build a baseline and a linear regression model.
- Interpret coefficients with appropriate caveats.
- Calculate and interpret residuals and regression metrics.
- Recognize overfitting and the need for held-out evaluation.

## 1. What is regression?

Regression predicts a numerical quantity, such as a price, temperature, or demand. Input variables are features (X); the value to predict is the target (y). Start with a simple baseline before evaluating a complex model.

```python
import numpy as np
X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)
```

## 2. Baseline models

A baseline provides a reference point. Predicting the training-set mean is a simple baseline for squared-error regression. A useful model should outperform a baseline on data it did not train on.

```python
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
print(mean_absolute_error(y_test, baseline.predict(X_test)))
```

## 3. Linear regression

Simple linear regression predicts ŷ = b₀ + b₁x. The intercept b₀ is the predicted value when x is zero; the slope b₁ describes the predicted change in y for a one-unit increase in x. Multiple linear regression uses several features. Coefficients are associations under the fitted model, not automatically causal effects.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
```

## 4. Residuals and loss

A residual is y − ŷ. Mean absolute error (MAE) averages absolute residuals. Mean squared error (MSE) averages squared residuals and penalizes large errors more strongly. RMSE is the square root of MSE and returns to the target's units.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
residuals = y_test - predictions
print(mae, mse, rmse, residuals)
```

## 5. Train and test data

Fit model parameters using training data. Use validation data or cross-validation to choose settings. Keep the test set for final evaluation. If you repeatedly tune based on test performance, the test set becomes part of the selection process and its score becomes optimistic.

```python
# Never fit the model on X_test or use y_test to choose model settings.
# For larger datasets, use a validation set or cross-validation.
```

## 6. Assumptions and diagnostics

Linear regression can be useful even when assumptions are imperfect, but interpretation and uncertainty depend on them. Inspect residuals versus predictions, nonlinearity, unequal variance, influential points, and correlated predictors. Extrapolation beyond the observed range can be unreliable.

```python
import matplotlib.pyplot as plt
plt.scatter(predictions, residuals)
plt.axhline(0)
plt.xlabel("Predicted values")
plt.ylabel("Residuals (actual - predicted)")
plt.show()
```

## 7. Regularization and overfitting

Overfitting occurs when a model captures training-specific noise and generalizes poorly. Ridge regression adds an L2 penalty; Lasso adds an L1 penalty and can shrink some coefficients to zero. Select regularization strength using validation data or cross-validation.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
ridge.fit(X_train, y_train)
print(ridge.predict(X_test))
```

## 8. Mini-project

Compare a mean baseline with linear regression using the same split. Report MAE and RMSE, explain what one unit of error means, and state limitations. For real datasets, split before fitting any data-dependent preprocessing.

## Mini-project

Complete this baseline implementation, run it, and explain the results. Change one design choice and describe how the output changes.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Synthetic, deterministic data for practice only
rng = np.random.default_rng(12)
X = np.arange(100, dtype=float).reshape(-1, 1)
y = 4.0 * X[:, 0] + 7 + rng.normal(0, 12, size=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print("Slope:", model.coef_[0])
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
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
