# 🍷 Wine Classification using Machine Learning

## 📌 Project Overview

This project involves building and evaluating machine learning models to classify wines into one of three cultivars based on their chemical composition. The dataset is sourced from the `sklearn.datasets.load_wine` dataset, which contains 178 samples and 13 numerical features.

The objective is to support a wine research institute in Italy in automating the classification process, identifying the most important chemical properties, and reducing the number of costly lab tests required.

---

## 🎯 Objectives

- Identify the most influential chemical features that differentiate wine cultivars.
- Determine if accurate classification is possible using fewer features (cost-saving initiative).
- Compare the performance of different classification algorithms.
- Visualize the separation between wine types using 2D dimensionality reduction (PCA).
- Recommend the best-performing model for deployment.

---

## 📂 Dataset Description

- **Samples**: 178
- **Features**: 13 chemical measurements (e.g., alcohol, flavanoids, proline)
- **Target**: 3 wine cultivars (Class 0, 1, 2)
- **Source**: `sklearn.datasets.load_wine`

---

## 🧪 Methodology

1. **Data Preprocessing**:
   - No missing values
   - StandardScaler applied to normalize feature values

2. **Feature Selection**:
   - Used Recursive Feature Elimination (RFE) with Logistic Regression & Random Forest
   - Identified top features like `flavanoids`, `proline`, `OD280/OD315`, and `color_intensity`

3. **Model Training & Evaluation**:
   - Trained 7 classifiers:
     - K-Nearest Neighbors (KNN)
     - Logistic Regression (with and without RFE)
     - Decision Tree
     - Random Forest
     - Support Vector Machine (SVM)
     - PCA + Logistic Regression
   - Used train-test split (80/20) and 5-fold cross-validation
   - Evaluated using precision, recall, F1-score, and accuracy

4. **Dimensionality Reduction**:
   - Applied PCA to visualize wine cultivar separation in 2D

---

## 📊 Results Summary

- **Top-performing models**: KNN, SVM, and full Logistic Regression, all with **98% accuracy**
- **Feature reduction (RFE)** maintained high accuracy (96%), validating the cost-saving goal
- **PCA visualization** revealed clear clustering between wine types, supporting model reliability
- **Random Forest** underperformed slightly, achieving 93% accuracy

---

## ✅ Recommendations

- Deploy **SVM or Logistic Regression** for automated classification in quality control
- Reduce testing to top 4–5 features (e.g., `flavanoids`, `proline`) for cost efficiency
- Use PCA plots for internal communication and reporting
- Schedule regular model retraining to maintain accuracy over time

---

## 📎 Dependencies

- Python 3.8+
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pandas`
- `numpy`

Install via:
```bash
