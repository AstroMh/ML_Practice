# Machine Learning Practice Projects (Python)

This repository contains my personal machine learning and computer vision practice projects in Python.  
The goal is to build a solid **basic–intermediate** understanding of ML by implementing algorithms from scratch or with popular libraries, experimenting on real datasets, and keeping everything in clean, reproducible `.py` scripts (instead of only Jupyter notebooks).

---

## 🔍 What’s inside

- **Self-contained ML projects**  
  Each project lives in its own folder with:
  - A main Python script (converted from Jupyter notebooks)
  - A small dataset (CSV or similar) if needed
  - Optional notes or results

- **Topics covered (so far / planned)**  
  This will grow over time, but includes things like:
  - k-Nearest Neighbors (KNN) classification
  - Data preprocessing & feature scaling
  - Basic data visualization (Matplotlib, Seaborn)
  - Train/test splits & model evaluation
  - Linear & logistic regression  
  - Decision trees & ensembles  
  - Unsupervised learning (k-Means, clustering, PCA)  
  - Intro computer vision / image-based models  

---

## 📂 Projects Index

A quick overview of the mini-projects in this repo.

| # | Project | Type | Folder | Main Script | Dataset | Short Description |
|---|---------|------|--------|------------|---------|-------------------|
| 1 | KNN Classification (Tabular) | Classification | [`projects/knn_classification/`](projects/knn_classification/) | `knn_project.py` | `KNN_Project_Data` (CSV) | Predict a binary target using K-Nearest Neighbors, with scaling, tuning over K, and evaluation plots. |
| 2 | Linear Regression – Ecommerce Customers | Regression | [`projects/linear_regression_ecommerce/`](projects/linear_regression_ecommerce/) | `linear_regression_project.py` | `Ecommerce Customers` (CSV) | Predict yearly customer spend from behavior/features using linear regression and residual analysis. |
| 3 | KNN – Digits (mini-MNIST) | Classification | [`projects/knn_digits/`](projects/knn_digits/) | `knn_digits_project.py` | `sklearn.datasets.load_digits` | Classify 8×8 handwritten digit images (0–9) with KNN, tuning K and visualizing predictions. |
| 4 | Linear Regression – California Housing | Regression | [`projects/california_housing_linear_regression/`](projects/california_housing_linear_regression/) | `california_housing_linear_regression.py` | `sklearn.datasets.fetch_california_housing` | Predict median house value in California districts with linear regression, including coefficient analysis. |
| 5 | Logistic Regression – Advertising Clicks | Classification | [`projects/advertising_logistic_regression/`](projects/advertising_logistic_regression/) | `advertising_logistic_regression.py` | `advertising.csv` (synthetic) | Predict whether a user will click on an online ad based on demographics and usage behavior. |

> 📝 Each project folder contains:
> - A main Python script (`*.py`)  
> - A `results/` directory with plots & metric reports  
> - A local `README.md` describing the project and its outputs

