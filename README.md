# 3rd-yr-lab-activities

A collection of practical laboratory notebooks and exercises completed during my 3rd year in **BS Data Science & Analytics, University of Santo Tomas.**

## Project Index

## 1. Machine Learning

| Folder                                              | Focus                                  | Description                                                                                                                                                                                                                                                                                                      |
|-----------------------------------------------------|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`Classification Metrics and ROCAUC/`**            | Classification Evaluation              | Implements confusion‐matrix construction, accuracy, per‐class precision/recall, macro‐averaging (precision, recall, F1), label‐binarization, ROC‐curve computation, AUC via trapezoidal rule, and plotting multiclass ROC‐AUC—all built from scratch using NumPy and Matplotlib.                                        |
| **`Regression Metrics and Preprocessing/`**          | Regression Metrics & Preprocessing      | Defines functions for Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² (coefficient of determination). Includes data standardization (z‐score) and min–max normalization routines, plus a one‐hot encoding utility for categorical labels. All code is handcrafted without relying on high‐level libraries. |
| **`Error Based Learning (Linear Regression)/`**     | Linear Regression & Error Analysis      | Uses the Wine Quality dataset to load and merge red/white CSVs, split 80/20 by hand, compute Sum of Squared Errors (SSE), initialize weights to zero, and train a linear regression model via gradient descent (with SSE‐based stopping criteria). Also standardizes features and evaluates final SSE on a held‐out test set.     |
| **`Similarity Based Learning for Classification/`** | k‐Nearest Neighbors & Distance Metrics   | Implements k‐Nearest Neighbors classification (Euclidean, Manhattan, Minkowski, Cosine) from scratch. Loads and cleans the Mushroom dataset (mapping categorical codes to labels), computes class distribution and Proportional Chance Criterion (PCC), evaluates correlation among features, and trains/evaluates KNN.                                         |

> **Note:** This repository is organized by topic. Each folder contains a `.ipynb` file and may include supporting datasets or resources.

## Purpose
To document and demonstrate my technical growth and understanding of machine learning fundamentals through lab work, in preparation for future internships or research.

## How to Use
Explore the folders above to view each lab's corresponding notebook and output. All notebooks were written using Python, NumPy, Pandas, Matplotlib, and other native libraries (no sklearn, unless otherwise specified).

## Note
This repository is educational and reflects learning progress. Some code may be experimental or intentionally implemented from scratch to reinforce understanding of machine learning principles.
