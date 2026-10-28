# Python Fundamentals for AI and Machine Learning

## Overview

Python is widely used in artificial intelligence because it has readable syntax and a large ecosystem of libraries for data processing, visualization, and machine learning. This lesson covers the Python fundamentals needed to work with datasets and implement machine-learning workflows.

**Estimated time:** 2–3 hours  
**Difficulty:** Beginner  
**Prerequisites:** Basic arithmetic and logical reasoning

## Learning objectives

By the end of this lesson, you should be able to:

1. Store and manipulate values using variables and data types.
2. Use conditions and loops to control program execution.
3. Work with lists, tuples, dictionaries, and sets.
4. Write reusable functions.
5. Process collections using loops and comprehensions.
6. Read and interpret common Python errors.
7. Write a small program that calculates statistics from a dataset.

## 1. Variables and data types

A variable is a name that refers to a value.

```python
student_name = "Aarav"
age = 16
accuracy = 0.92
is_ready = True

print(student_name)
print(age)
print(accuracy)
print(is_ready)
```

| Type | Example | Purpose |
|---|---|---|
| `int` | `16` | Whole numbers |
| `float` | `0.92` | Decimal numbers |
| `str` | `"AI"` | Text |
| `bool` | `True` | Logical values |
| `list` | `[4, 7, 9]` | Ordered, mutable collection |
| `tuple` | `(4, 7, 9)` | Ordered, immutable collection |
| `dict` | `{"score": 92}` | Key-value mapping |
| `set` | `{4, 7, 9}` | Collection of unique values |

Check a value's type with `type()`.

```python
score = 92.5
print(type(score))  # <class 'float'>
```

### Type conversion

```python
number_text = "42"
number = int(number_text)
print(number + 8)  # 50
```

Conversion only works when the value can be represented in the requested type. For example, `int("hello")` raises a `ValueError`.

**AI connection:** Dataset columns can contain integers, decimal measurements, text labels, or Boolean indicators. Understanding types helps prevent incorrect calculations and data-processing errors.

## 2. Operators and expressions

Arithmetic operators include `+`, `-`, `*`, `/`, `//`, `%`, and `**`.

```python
print(10 + 3)   # 13
print(10 / 3)   # 3.333...
print(10 // 3)  # 3
print(10 % 3)   # 1
print(2 ** 4)   # 16
```

Comparison operators return Boolean values:

```python
score = 85
print(score >= 80)  # True
print(score == 100) # False
print(score != 0)   # True
```

Logical operators combine conditions:

```python
score = 85
attendance = 90
eligible = score >= 80 and attendance >= 75
print(eligible)  # True
```

Use `==` to compare values and `=` to assign a value.

## 3. Conditional statements

Conditions allow a program to make decisions.

```python
accuracy = 0.91

if accuracy >= 0.95:
    print("Excellent")
elif accuracy >= 0.85:
    print("Good")
else:
    print("Needs improvement")
```

Python uses indentation to define blocks. Statements belonging to an `if`, `elif`, `else`, loop, or function must be indented consistently.

Conditions can also check membership:

```python
label = "spam"
if label in {"spam", "fraud"}:
    print("Flag for review")
```

**AI connection:** A program may use conditions to flag suspicious transactions or decide whether a model score exceeds a chosen threshold. The rule itself is not necessarily a trained machine-learning model.

## 4. Loops

### `for` loops

Use a `for` loop to process each item in a collection.

```python
scores = [72, 85, 91, 68]
for score in scores:
    print(score)
```

Use `range()` when you need a sequence of integers.

```python
for i in range(5):
    print(i)
```

This prints `0` through `4`, not `1` through `5`.

### `while` loops

A `while` loop repeats while its condition remains true.

```python
count = 0
while count < 3:
    print(count)
    count += 1
```

Make sure a `while` loop has a way to stop. Otherwise, it may run indefinitely.

### Accumulation

A common pattern is to maintain a running total.

```python
scores = [72, 85, 91, 68]
total = 0
for score in scores:
    total += score
print(total)  # 316
```

## 5. Lists and indexing

Lists store ordered collections. Indexing begins at zero.

```python
scores = [72, 85, 91, 68]
print(scores[0])   # 72
print(scores[-1])  # 68
print(scores[1:3]) # [85, 91]
```

Useful list operations:

```python
scores.append(88)
scores.remove(72)
print(len(scores))
print(sorted(scores))
```

`sorted(scores)` returns a sorted list without changing the original list. `scores.sort()` sorts the original list in place.

### List comprehensions

```python
scores = [72, 85, 91, 68]
passed = [score for score in scores if score >= 75]
print(passed)  # [85, 91]
```

The equivalent loop is:

```python
passed = []
for score in scores:
    if score >= 75:
        passed.append(score)
```

Use comprehensions for simple transformations. A regular loop is often easier to read when the logic is complex.

## 6. Dictionaries, tuples, and sets

### Dictionaries

A dictionary maps unique keys to values.

```python
student = {
    "name": "Aarav",
    "score": 92,
    "subject": "AI"
}

print(student["score"])
student["score"] = 95
student["grade"] = "A"
print(student.get("school", "Unknown"))
```

Use `.get()` when a key may be missing; it returns the provided default rather than raising a `KeyError`.

### Tuples

Tuples are ordered collections that cannot be modified after creation.

```python
point = (3, 7)
x, y = point
print(x, y)
```

### Sets

Sets store unique values and support membership tests and set operations.

```python
labels = {"cat", "dog", "cat"}
print(labels)  # Contains only one "cat"
print("dog" in labels)
```

Sets are useful when checking unique labels or removing duplicates. They do not provide a meaningful positional index for data analysis.

## 7. Functions

Functions package reusable logic.

```python
def calculate_mean(values):
    if len(values) == 0:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)

scores = [72, 85, 91, 68]
print(calculate_mean(scores))
```

Parts of a function:

- `def` starts the function definition.
- `calculate_mean` is the function name.
- `values` is a parameter.
- `return` sends the result back to the caller.

### Parameters and return values

```python
def percentage(correct, total):
    if total <= 0:
        raise ValueError("total must be positive")
    return (correct / total) * 100

print(percentage(18, 20))  # 90.0
```

A function should ideally perform a clearly defined task. Small functions are easier to test and reuse.

### Scope

A variable created inside a function is normally local to that function.

```python
def double(number):
    result = number * 2
    return result

print(double(6))  # 12
```

The local variable `result` is not available outside the function unless returned or otherwise exposed.

## 8. Working with collections of measurements

Consider a small dataset of crop temperatures.

```python
temperatures = [28.5, 30.0, 29.5, 31.0, 27.0]

mean_temperature = sum(temperatures) / len(temperatures)
minimum_temperature = min(temperatures)
maximum_temperature = max(temperatures)

print("Mean:", mean_temperature)
print("Minimum:", minimum_temperature)
print("Maximum:", maximum_temperature)
```

This illustrates how basic Python can summarize measurements before using a library such as NumPy or pandas.

### Counting categories

```python
labels = ["healthy", "diseased", "healthy", "healthy", "diseased"]
counts = {}

for label in labels:
    counts[label] = counts.get(label, 0) + 1

print(counts)
```

Expected result:

```text
{'healthy': 3, 'diseased': 2}
```

The dictionary accumulates one count for each observed category.

## 9. Errors and debugging

| Error | Typical cause |
|---|---|
| `SyntaxError` | Invalid Python syntax |
| `NameError` | Using an undefined name |
| `TypeError` | An operation is incompatible with a value's type |
| `ValueError` | A value has the wrong form for an operation |
| `IndexError` | Accessing an invalid list index |
| `KeyError` | Accessing a missing dictionary key |
| `ZeroDivisionError` | Dividing by zero |

Example:

```python
scores = [70, 80]

# This causes IndexError:
# print(scores[5])

index = 1
if 0 <= index < len(scores):
    print(scores[index])
```

Debugging method:

1. Read the final line of the traceback.
2. Find the line of your code mentioned in the traceback.
3. Check variable names, types, indexes, and assumptions.
4. Reproduce the smallest example that causes the error.
5. Fix the cause rather than merely hiding the error.

## 10. Mini-project: Dataset summary

Write a program that receives a list of numerical scores and reports the count, mean, minimum, maximum, and number of scores at least 75.

```python
def summarize_scores(scores):
    if not scores:
        raise ValueError("scores must not be empty")

    total = sum(scores)
    mean = total / len(scores)
    passed = sum(1 for score in scores if score >= 75)

    return {
        "count": len(scores),
        "mean": mean,
        "minimum": min(scores),
        "maximum": max(scores),
        "at_least_75": passed
    }

scores = [72, 85, 91, 68, 77]
summary = summarize_scores(scores)

for key, value in summary.items():
    print(f"{key}: {value}")
```

Expected results:

- Count: `5`
- Mean: `78.6`
- Minimum: `68`
- Maximum: `91`
- Scores at least 75: `3`

### Why this matters for IOAI

Many AI tasks begin with raw examples and labels. You must be able to inspect values, count categories, compute simple summaries, and transform data before using machine-learning algorithms.

## Key takeaways

- Variables hold references to values; types affect which operations are valid.
- Conditions and loops express decisions and repetition.
- Lists, dictionaries, tuples, and sets serve different purposes.
- Functions make code reusable and testable.
- Errors provide clues about violated assumptions.
- Basic data summaries are the starting point for data science.

## Recommended resources

- Python Tutorial: https://docs.python.org/3/tutorial/
- IOAI Education Hub: https://ioai-official.org/resources/
- Kaggle Learn: https://www.kaggle.com/learn
- Google Machine Learning Crash Course prerequisites: https://developers.google.com/machine-learning/crash-course/prereqs-and-prework
