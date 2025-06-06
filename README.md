# 3rd Year Lab Activities

A collection of laboratory notebooks and exercises completed during my 3rd year in **BS Data Science & Analytics, University of Santo Tomas.**

## Project Index

### 1. Machine Learning
| Folder | Focus | Description |
| ------ | ----- | ----------- |
| [**`Classification Metrics and ROCAUC/`**](./Machine%20Learning/Classification%20Metrics%20and%20ROCAUC) | Classification Evaluation | Confusion‐matrix, accuracy, per‐class precision/recall, macro‐averaging (precision, recall, F1), label‐binarization, ROC‐curve computation, AUC via trapezoidal rule, and plotting multiclass ROC‐AUC. Built using NumPy and Matplotlib. |
| [**`Error Based Learning/`**](./Machine%20Learning/Error%20Based%20Learning%20(Linear%20Regression)) | Linear Regression & Error Analysis | Uses the Wine Quality dataset to train a linear regression model via gradient descent. Includes feature standardization, SSE calculation, weight initialization, and test set evaluation. All done from scratch without sklearn. |
| [**`Regression Metrics and Preprocessing/`**](./Machine%20Learning/Regression%20Metrics%20and%20Preprocessing) | Regression Metrics & Preprocessing | Defines MSE, MAE, and R², performs feature standardization (z‐score), min–max normalization, and one‐hot encoding of labels. All implemented using NumPy. |
| [**`Similarity Based Learning/`**](./Machine%20Learning/Similarity%20Based%20Learning%20for%20Classification%20and%20Recommender%20System) | KNN & Recommender Systems | Trains a k‐Nearest Neighbors classifier on the Mushroom dataset using different distance metrics (Euclidean, Manhattan, Cosine, Minkowski). Evaluates correlations and accuracy. Also implements a recommender using MovieLens user-item cosine similarity. |

> **Note:** Each folder contains a `.ipynb` notebook and may include supplementary datasets or files.
<br>

### 2. Principles of Big Data

| Folder | Focus | Description |
| ------ | ----- | ----------- |
| [**`Clustering/`**](./Principles%20of%20Big%20Data/Clustering) | Customer Segmentation & Clustering | Loads “OnlineRetail.csv,” cleans data, computes RFM (Recency, Frequency, Monetary), removes outliers (IQR), scales features, finds optimal K using Elbow and Silhouette, applies K-Means (K=3), and visualizes clusters. |
| [**`Predictive Modeling & Bayesian Methods/`**](./Principles%20of%20Big%20Data/Predictive%20Modeling%20%26%20Bayesian%20Methods) | Supervised Learning & Bayesian Inference | Performs EDA on a synthetic loan dataset. Trains Gaussian Naïve Bayes, custom Bayesian Network, and tuned k-NN with 5-fold stratified CV. Includes logistic regression and Bayesian network primer. |
| [**`Time Series Forecasting/`**](./Principles%20of%20Big%20Data/Time%20Series%20Forecasting) | Time-Series Analysis & Forecasting | Pulls AMP stock data via yfinance, checks stationarity (ADF, KPSS), differences to d=1, selects ARIMA/SARIMAX via grid search (AIC), evaluates model residuals and forecasts 30 business days ahead. |
> **Note:** Each folder contains a `.ipynb` notebook and may include supplementary datasets or files.

## Purpose
To document and demonstrate my technical growth and understanding of machine learning fundamentals through lab work, in preparation for future internships or research.

## How to Use
Explore the folders above to view each lab’s Jupyter notebook. All code was written manually with libraries like NumPy, Pandas, and Matplotlib (unless otherwise noted).

## Note
This repository is educational and reflects learning progress. Some code may be experimental or simplified to strengthen conceptual understanding.
