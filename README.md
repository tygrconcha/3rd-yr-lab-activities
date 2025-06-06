# 3rd Year Lab Activities

A collection of practical laboratory notebooks and exercises completed during my 3rd year in **BS Data Science & Analytics, University of Santo Tomas.**

## Project Index

### 1. Machine Learning
| Folder | Focus | Description |
| ------ | ----- | ----------- |
| [**`Classification Metrics and ROCAUC/`**](./Machine%20Learning/Classification%20Metrics%20and%20ROCAUC) | Classification Evaluation | Confusion‐matrix, accuracy, per‐class precision/recall, macro‐averaging (precision, recall, F1), label‐binarization, ROC‐curve computation, AUC via trapezoidal rule, and plotting multiclass ROC‐AUC—built from scratch using NumPy and Matplotlib. |
| [**`Error Based Learning (Linear Regression)/error_based_learning_(linear_regression).ipynb`**](./Machine%20Learning/Error%20Based%20Learning%20(Linear%20Regression)/error_based_learning_(linear_regression).ipynb) | Linear Regression & Error Analysis | Uses the Wine Quality dataset to load/merge red & white CSVs, split 80/20, compute Sum of Squared Errors (SSE), initialize weights to zero, and train a linear regression model via gradient descent (with SSE‐based stopping criteria). Standardizes features and evaluates final SSE on a held‐out test set. |
| [**`Regression Metrics and Preprocessing/regression_metrics_and_preprocessing.ipynb`**](./Machine%20Learning/Regression%20Metrics%20and%20Preprocessing/regression_metrics_and_preprocessing.ipynb) | Regression Metrics & Preprocessing | Defines functions for Mean Squared Error (MSE), Mean Absolute Error (MAE), and R². Includes data standardization (z‐score) and min–max normalization routines, plus a one‐hot encoding utility for categorical labels—hand‐coded without high‐level libraries. |
| [**`Similarity Based Learning for Classification and Recommender System/similarity_based_learning_for_classification_and_recommender_system.ipynb`**](./Machine%20Learning/Similarity%20Based%20Learning%20for%20Classification%20and%20Recommender%20System/similarity_based_learning_for_classification_and_recommender_system.ipynb) | k‐Nearest Neighbors & Recommender System | Implements k‐Nearest Neighbors classification (Euclidean, Manhattan, Minkowski, Cosine) on the Mushroom dataset (with mapping of categorical codes), computes class distribution and Proportional Chance Criterion (PCC), evaluates feature correlations, and trains/evaluates KNN. Also builds a simple user‐item recommender on MovieLens. |

> **Note:** This repository is organized by topic. Each folder contains a `.ipynb` file and may include supporting datasets or resources.

## Purpose
To document and demonstrate my technical growth and understanding of machine learning fundamentals through lab work, in preparation for future internships or research.

## How to Use
Explore the folders above to view each lab's corresponding notebook and output. All notebooks were written using Python, NumPy, Pandas, Matplotlib, and other native libraries (no sklearn, unless otherwise specified).

## Note
This repository is educational and reflects learning progress. Some code may be experimental or intentionally implemented from scratch to reinforce understanding of machine learning principles.
