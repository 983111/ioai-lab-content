# Reproducible ML Experiments

## Overview
Run controlled comparisons, record configurations, and make results easier to reproduce.

**Time:** 30–45 minutes · **Level:** Intermediate

## Learning objectives
- Fix random seeds and understand their limits.
- Establish a baseline.
- Change one variable at a time.
- Record configurations and metrics.

## Workflow
Generate a fixed synthetic dataset, split it repeatably, define a shared evaluation function, sweep one regularization parameter, and save a CSV results table.

## Key idea
A fair experiment changes one factor while holding others constant. Record dataset settings, preprocessing, model parameters, seed, package versions, and metrics. Seeds do not guarantee bit-for-bit reproducibility across all environments.

## Exercises
Repeat with multiple seeds, compare class weighting, and explain why simultaneous changes make results difficult to interpret.