# 3rd Year Lab Activities

A collection of laboratory notebooks and exercises completed during my 3rd year in **BS Data Science & Analytics, University of Santo Tomas.**

## Project Index

### 1. Machine Learning
| Folder | Focus | Description |
| ------ | ----- | ----------- |
| [**`Classification Metrics and ROCAUC/`**](./Machine%20Learning/Classification%20Metrics%20and%20ROCAUC) | Classification Evaluation | Implements key classification evaluation tools such as confusion matrix, accuracy, precision, recall, and F1-score (both per-class and macro-averaged). Also builds ROC curves and calculates AUC using the trapezoidal rule using NumPy and Matplotlib. |
| [**`Error Based Learning/`**](./Machine%20Learning/Error%20Based%20Learning%20(Linear%20Regression)) | Linear Regression & Error Analysis | Builds a linear regression model from scratch using the Wine Quality dataset. Includes steps for standardizing features, initializing weights, computing prediction errors (SSE), and evaluating test performance without using sklearn. |
| [**`Regression Metrics and Preprocessing/`**](./Machine%20Learning/Regression%20Metrics%20and%20Preprocessing) | Regression Metrics & Preprocessing | Defines and implements regression metrics like MSE, MAE, and R². Also performs z-score standardization, min-max normalization, and one-hot encoding using only NumPy for full control and learning. |
| [**`Similarity Based Learning/`**](./Machine%20Learning/Similarity%20Based%20Learning%20for%20Classification%20and%20Recommender%20System) | KNN & Recommender Systems | Trains a k-Nearest Neighbors classifier on the Mushroom dataset using multiple distance metrics (Euclidean, Manhattan, Cosine, Minkowski). Also builds a movie recommender system using user-item cosine similarity from the MovieLens dataset. |

> **Note:** Each folder contains a `.ipynb` notebook and may include supplementary datasets or files.
<br>

### 2. Principles of Big Data

| Folder | Focus | Description |
| ------ | ----- | ----------- |
| [**`Clustering/`**](.Principles%20of%20Big%20Data/Clustering) | Customer Segmentation & Clustering | Processes online retail data, computes RFM metrics, removes outliers, scales features, and uses the Elbow and Silhouette methods to find the best K. Applies K-Means (K=3) to segment customers and visualizes the results. |
| [**`Predictive Modeling & Bayesian Methods/`**](.Principles%20of%20Big%20Data/Predictive%20Modeling%20and%20Bayesian%20Methods) | Supervised Learning & Bayesian Inference | Explores a synthetic loan dataset and trains three models: Gaussian Naïve Bayes, a custom Bayesian Network, and tuned k-NN. Evaluates performance with 5-fold CV and includes a logistic regression demo and BN primer. |
| [**`Time Series Forecasting/`**](.Principles%20of%20Big%20Data/Time%20Series%20Forecasting) | Time-Series Analysis & Forecasting | Downloads historical stock prices, checks for stationarity (ADF, KPSS), applies differencing, fits ARIMA and SARIMAX models via AIC-based grid search, and forecasts future values with residual checks. |
> **Note:** Each folder contains a `.ipynb` notebook and may include supplementary datasets or files.

## Purpose
This repository showcases key topics in machine learning and big data principles. The goal is to document my applied skills and analytical thinking through structured, hands-on tasks that reinforce both theoretical and practical competencies. It also serves as a reference portfolio for internships, capstone preparation, or peer collaboration.

## How to Use
Each section contains folders with clearly labeled Jupyter notebooks (`.ipynb`) and supporting files. Browse the tables above to explore topics by focus area. You can open each notebook to review code, visualizations, and embedded insights.
## Note
This repository is educational and reflects learning progress. Some code may be experimental or simplified to strengthen conceptual understanding.
