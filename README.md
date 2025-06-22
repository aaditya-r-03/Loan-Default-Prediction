# Loan Default Prediction using Machine Learning

This project focuses on predicting loan defaulters using various machine learning techniques, starting from a basic model and improving it with advanced preprocessing, feature engineering, and interpretability techniques.

## 🔍 Problem Statement
Loan default prediction is a key problem in the financial sector. This project aims to build a model that accurately predicts defaulters based on borrower data while addressing challenges like class imbalance, feature selection, and explainability.

---

## 📁 Project Structure
- `baseline_model.ipynb` — A simple Random Forest model with basic preprocessing and one-hot encoding.
- `updated_model.ipynb` — Improved version using XGBoost, SMOTE, target encoding, isolation forest, cost-sensitive learning, and SHAP for interpretability.
- `dataset/` — Contains the input CSV file used for training and testing.

---

## 🧠 Key Techniques Used
- Target Encoding for categorical features
- Isolation Forest for outlier removal
- SMOTE for handling class imbalance
- XGBoost with regularization and early stopping
- Cost-sensitive learning using sample weights
- SHAP for model explainability

---

## 📊 Evaluation Metrics

### 🔹 Baseline Model (Random Forest)
| Metric        | Value |
|---------------|--------|
| **Accuracy**      | 88.6%  |
| **Precision (Class 1 - Defaulters)** | 60% |
| **Recall (Class 1 - Defaulters)**    | 5%  |
| **F1-Score (Class 1)**               | 9%  |
| **Macro Avg F1-Score**              | 52% |
| **Weighted Avg F1-Score**           | 84% |

### 🔹 Final Model (XGBoost + SMOTE + SHAP)
| Metric        | Value |
|---------------|--------|
| **Accuracy**      | 91%    |
| **Precision (Class 1 - Defaulters)** | 98% |
| **Recall (Class 1 - Defaulters)**    | 74% |
| **F1-Score (Class 1)**               | 84% |
| **Macro Avg F1-Score**              | 89% |
| **Weighted Avg F1-Score**           | 90% |

> 📌 *The improved model achieved a **~69% increase in recall** and significantly boosted F1-score for defaulters (Class 1), while maintaining high precision.*



---

## 🛠️ Libraries Used
- `pandas`, `numpy`, `sklearn`, `xgboost`, `imblearn`, `shap`, `matplotlib`, `seaborn`

---

## ✍️ Author
Aaditya Roshan
