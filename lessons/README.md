# IOAI Lab Lessons

This directory contains a progressive, practical curriculum for Python and machine learning. Lesson content is stored as Markdown, quizzes and exercises as JSON, and executable practice as Jupyter notebooks.

## Lesson format

Each lesson folder should contain:

- `lesson.md`: explanations, formulas where relevant, worked examples, AI connections, common mistakes, and further reading.
- `quiz.json`: multiple-choice questions with answer keys and explanations.
- `exercises.json`: coding tasks, requirements, starter code, hints, test cases, and learning outcomes.
- `starter.ipynb`: a notebook suitable for Jupyter and Google Colab.

## Learning sequence

| Order | Folder | Main topics | Expected outcome |
|---|---|---|---|
| 1 | `python-fundamentals/` | Variables, types, operators, conditions, loops, collections, functions, debugging | Write small, testable Python programs and summarize simple data |
| 2 | `data-manipulation/` | NumPy arrays, pandas DataFrames, selection, filtering, missing values, encoding, scaling, visualization | Clean and prepare a dataset |
| 3 | `statistics/` | Mean, median, mode, variance, standard deviation, percentiles, probability, distributions, sampling, correlation | Describe data and reason about variation |
| 4 | `regression/` | Features and targets, baselines, linear regression, coefficients, residuals, MAE, MSE, RMSE, overfitting | Train a model to predict numerical values |
| 5 | `classification/` | Labels, binary and multiclass classification, logistic regression, K-NN, decision trees, probabilities, thresholds | Train a model to predict categories |
| 6 | `model-evaluation/` | Train/validation/test splits, cross-validation, confusion matrix, precision, recall, F1, ROC-AUC, leakage, class imbalance | Evaluate generalization on unseen data |

## Current status

- [x] `python-fundamentals/lesson.md`
- [x] `python-fundamentals/quiz.json`
- [x] `python-fundamentals/exercises.json`
- [x] `python-fundamentals/starter.ipynb`
- [x] `data-manipulation/lesson.md`, `quiz.json`, `exercises.json`, `starter.ipynb`
- [x] `statistics/lesson.md`, `quiz.json`, `exercises.json`, `starter.ipynb`
- [x] `regression/lesson.md`, `quiz.json`, `exercises.json`, `starter.ipynb`
- [x] `classification/lesson.md`, `quiz.json`, `exercises.json`, `starter.ipynb`
- [x] `model-evaluation/lesson.md`, `quiz.json`, `exercises.json`, `starter.ipynb`

## Data conventions

- Use stable, lowercase, kebab-case lesson and question IDs.
- Keep quiz answers and explanations in the data so the frontend can render feedback consistently.
- Coding exercises should include explicit requirements and deterministic test cases where possible.
- Use synthetic or clearly attributed sample data in teaching examples. Do not present fabricated data as official IOAI competition data.
- Link external resources to their original providers and distinguish supplementary teaching material from official IOAI content.

## References

- IOAI Education Hub: https://ioai-official.org/resources/
- Python Tutorial: https://docs.python.org/3/tutorial/
- Kaggle Learn: https://www.kaggle.com/learn
- Google Machine Learning Crash Course: https://developers.google.com/machine-learning/crash-course/
- NumPy documentation: https://numpy.org/doc/stable/
- pandas documentation: https://pandas.pydata.org/docs/
- scikit-learn documentation: https://scikit-learn.org/stable/
