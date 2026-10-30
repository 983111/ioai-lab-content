# Image Classification with a Small CNN

## Overview
Train a compact convolutional neural network on CIFAR-10 with PyTorch. This is an educational baseline, not a state-of-the-art model.

**Time:** 60–90 minutes · **Level:** Intermediate

## Learning objectives
- Explain convolution, pooling, and feature maps.
- Prepare image batches and normalize channels.
- Train a small CNN with cross-entropy loss.
- Plot loss and accuracy to reason about overfitting.

## Workflow
1. Load CIFAR-10 using torchvision.
2. Inspect labelled images.
3. Define convolutional blocks and a classifier.
4. Train and evaluate for a few epochs.
5. Plot learning curves and discuss overfitting.

## Exercises
Change filter counts, create a validation split, and explain why the test set should not guide repeated tuning.

**Compute:** Colab GPU is recommended but optional. The first run downloads CIFAR-10.