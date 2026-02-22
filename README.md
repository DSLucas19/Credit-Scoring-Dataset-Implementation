# Credit-Scoring-Dataset-Implementation
A basic EDA + Modeling of a Credit Scoring Dataset
# 🏦 Bank Credit Scoring — EDA & Classification

> A machine learning project to predict whether a bank should approve a loan for a customer, with a focus on maximizing **Precision** to minimize costly lending mistakes.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Objectives](#project-objectives)
- [Workflow](#workflow)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Feature Importance](#feature-importance)
- [Technologies Used](#technologies-used)
- [Conclusion](#conclusion)

---

## Overview

This project performs an end-to-end Exploratory Data Analysis (EDA) and classification pipeline on a bank credit scoring dataset. The goal is to build a model that helps a bank identify which customers are likely to repay their loans — and which are not.

---

## Dataset

| Property | Details |
|---|---|
| **Source** | [Kaggle — Bank Credit Scoring](https://www.kaggle.com/datasets/kapturovalexander/bank-credit-scoring) |
| **File** | `bank.csv` (semicolon-separated) |
| **Target Column** | `y` — whether the customer received a loan (`yes` / `no`) |
| **Data Quality** | Clean — no missing values found |

---

## Project Objectives

- 🎯 **Primary Goal**: Maximize **Precision** — ensure the bank does not approve loans for customers unlikely to repay them.
- 📊 **Secondary Goal**: Understand which features drive loan approval decisions.

---

## Workflow

### 1. 🔍 Data Preprocessing

- Loaded the dataset and inspected its structure using `.describe()`, `.info()`, and `.isna().sum()`.
- Confirmed the data was already clean, requiring **no additional data cleaning**.

### 2. 📈 Exploratory Data Analysis (EDA)

Explored key features and their relationship with the target variable:

- **Housing** — distribution of customers with/without housing loans.
- **Marital Status** — breakdown across marital categories.
- **Month** — call month distributions split by loan outcome.
- **Duration** — analyzed call duration on a log scale; longer calls strongly correlate with a `yes` outcome.

> 💡 **Insight**: The dataset has a significant **class imbalance** — the `yes` class is underrepresented. This was addressed in feature engineering.

### 3. ⚙️ Feature Engineering

Two key transformations were applied to prepare the data for modeling:

**Label Encoding** — converted all categorical columns to numeric values:
```
job, marital, education, default, housing, loan, contact, month, poutcome, y
```

**Random Over Sampling (ROS)** — used to handle the class imbalance problem by resampling the minority class to balance the dataset.

**Min-Max Scaling** — normalized all features to the `[0, 1]` range to prevent large numerical values from dominating the model.

### 4. 🌲 Model — Random Forest Classifier

Data was split into training and test sets (90% / 10%) and a **Random Forest Classifier** was trained with the following hyperparameters:

| Hyperparameter | Value |
|---|---|
| `criterion` | entropy |
| `max_depth` | None |
| `min_samples_leaf` | 1 |
| `min_samples_split` | 4 |
| `n_estimators` | 200 |

---

## Model Performance

The trained model achieved an overall **accuracy of 98%**, with strong Precision scores across both classes — directly satisfying the project's primary objective.

---

## Feature Importance

After training, feature importances were extracted and ranked. The top finding:

> 📞 **`duration`** — the length of the customer's phone call — is the **most important predictor** of loan approval. The longer a customer stays on a call, the higher the probability they will receive a loan.

---

## Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Data visualization |
| `scikit-learn` | Model training, scaling, evaluation |
| `imbalanced-learn` | Handling class imbalance (ROS, SMOTE) |

---

## Conclusion

- The dataset required minimal preprocessing — no data cleaning was necessary.
- Encoding, oversampling, and scaling techniques were successfully applied to prepare the data.
- The **Random Forest Classifier** achieved **98% accuracy**, reliably predicting loan eligibility.
- **Call duration** emerged as the strongest signal: customers who stay on the phone longer are significantly more likely to take out a loan.

---

*Project by: EDA.ipynb*
