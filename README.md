# Predicting Loan Payback

## Overview
This project focuses on predicting the probability that a borrower will fully repay a loan using classical machine learning models.  
The task is formulated as a binary classification problem with probabilistic outputs, based on a tabular dataset from the Kaggle Playground Series.
This project includes:
A complete end-to-end ML workflow
Feature preprocessing using sklearn Pipelines
Model comparison across Logistic Regression, KNN, and Random Forest
Probability-based predictions

---

## Problem Statement
Given borrower demographic and financial information, the goal is to estimate the probability that a loan will be paid back in full.  
Model performance is evaluated using ROC-AUC, which measures the quality of probability-based predictions.

---

## Dataset
The dataset is provided by the Kaggle Playground Series: *Predicting Loan Payback*.
The original dataset used to generate this competition data is publicly available at:  
https://www.kaggle.com/datasets/nabhihazahid/loan-prediction-dataset-2025/data

**Files:**
- `train.csv`: training data with labels  
- `test.csv`: test data without labels  
- `sample_submission.csv`: example submission format  

The dataset contains a mix of numerical and categorical features and is designed to resemble a real-world loan prediction task.
**Numerical features**
- `annual_income`
- `debt_to_income_ratio`
- `credit_score`
- `loan_amount`
- `interest_rate`

**Categorical features**
- `gender`
- `marital_status`
- `education_level`
- `employment_status`
- `loan_purpose`
- `grade_subgrade`

The target variable is:
- `loan_paid_back` (binary label indicating whether the loan was repaid)

---

## Data Split & Preprocessing

### Data Split
- The dataset is split into **75% training** and **25% validation**
- Stratified sampling is used to preserve class balance
- The same split is applied to all models for fair comparison

### Preprocessing
All preprocessing steps are implemented using `sklearn` Pipelines and `ColumnTransformer` to avoid data leakage.

- **Numerical features**
  - Median imputation
  - Standardization
- **Categorical features**
  - Most-frequent imputation
  - One-hot encoding

---

## Models

### Logistic Regression
Logistic Regression is used as a baseline model.  
An L2-regularized formulation is applied to improve generalization in the presence of correlated and high-dimensional features.
Fast to train and interpret.

---

### K-Nearest Neighbors (KNN)
KNN is implemented as a non-parametric baseline model.  
To reduce computational cost, the model is trained on a stratified subset of the training data.  
Dimensionality reduction is applied using Truncated SVD.
Different values of *k* are evaluated.

---

### Random Forest
Random Forest is used as an ensemble-based model capable of capturing nonlinear feature interactions.  
Class imbalance is addressed using class-weighted training, and the model is trained on a stratified subset of the training data for efficiency.  
More robust to noise, imbalance, and high-dimensional OHE features.
Achieves the best ROC-AUC and accuracy.
This model achieves the best validation ROC-AUC and is selected as the final model.

---

## Evaluation
Models are evaluated on the validation set using:
- Accuracy
- ROC-AUC (primary metric)
- Confusion Matrix

Among all tested models, the Random Forest classifier achieves the strongest overall performance and is chosen for final prediction.

---

## Confusion Matrix Interpretation
Random Forest shows:
- High True Positives (In this case, it is correctly predicted successful repayments)
- Reasonable ability to detect defaults despite imbalance
- Lower False Positive Rate compared to KNN
Therefore, this suggests Random Forest is the most reliable model for risk-sensitive financial applications.

---

## Final Model & Test Prediction
The selected Random Forest model is used to generate predictions on the test set.  
The model outputs **probability scores** representing the likelihood that a loan will be paid back, which are formatted according to Kaggle submission requirements.

---

## Repository Structure

```
Predicting-Loan-Payback/
├── Predicting_Loan_Payback.ipynb
├── train.csv
├── test.csv
├── sample_submission.csv
└── README.md
```

---

## Conclusion
This project demonstrates a full ML workflow for tabular classification:
- Rigorous preprocessing with pipelines
- Multiple baseline and advanced models
- Comparative evaluation
- Handling of class imbalance
- Production of probability-based predictions
  
The use of pipelines ensures clean and reproducible experimentation, while multiple models provide insight into different modeling trade-offs. In this case, the Random Forest emerges as the strongest model, balancing predictive performance, robustness, and handling of complex patterns.

