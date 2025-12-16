# Predicting Loan Payback

## Overview
This project focuses on **predicting the probability that a borrower will pay back their loan** using classical machine learning models including **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Random Forest**.

This project includes:
- A complete end-to-end ML workflow
- Feature preprocessing using sklearn Pipelines
- Model comparison across Logistic Regression, KNN, and Random Forest
- Probability-based predictions

---

## Problem Statement

Accurately assessing whether a borrower will repay a loan is a critical task in real-world **credit risk management**. Poor repayment decisions can lead to financial losses for lenders, while overly conservative decisions may deny credit to qualified applicants.

In this project, we aim to predict the **probability that a borrower will fully repay a loan** using **demographic and financial features**, such as income, debt-to-income ratio, credit score, employment status, and loan characteristics. Rather than making a hard yes/no decision, the model outputs a **probabilistic estimate**, which is more informative for practical decision-making.

The task is formulated as a **binary classification problem with probabilistic outputs**, and model performance is evaluated using **ROC-AUC**, a metric that measures how well a model ranks borrowers by repayment risk. This setup reflects real-world lending scenarios, where **risk ranking** is often more important than exact classification accuracy.


---

## Dataset
The dataset is provided by the Kaggle Playground Series: *Predicting Loan Payback*.

All training and test data can be downloaded directly from the official Kaggle page:
https://www.kaggle.com/competitions/playground-series-s5e11/data

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

- The dataset is split into **75% training** and **25% validation** using stratified sampling to preserve class balance.
- The same train/validation split is used for all models to ensure fair comparison.
- For efficiency, subset sampling is applied only to the training data:
  - Logistic Regression uses the full training set.
  - KNN is trained on a 10% stratified subset of the training data.
  - Random Forest is trained on a 15% stratified subset of the training data.
- All models are evaluated on the same full validation set.

### Preprocessing

All preprocessing steps are implemented using `sklearn` Pipelines and `ColumnTransformer` to avoid data leakage. The preprocessing pipeline is fitted on the training data only and applied consistently across all models.

- **Numerical features**
  - Median imputation
  - Standardization
- **Categorical features**
  - Most-frequent imputation
  - One-hot encoding (with unseen categories handled safely)



---

## Models

### Logistic Regression

- Logistic Regression is used as a **baseline classification model**.  
- An **L2-regularized** formulation is applied to improve generalization for high-dimensional one-hot encoded features.  
- The model is trained using a unified preprocessing pipeline and serves as a fast and interpretable baseline.


### K-Nearest Neighbors (KNN)

- K-Nearest Neighbors (KNN) is used as a non-parametric baseline model.  
- To reduce computational cost, the model is trained on a **10% stratified subset** of the training data.  
- **Truncated SVD** is applied for dimensionality reduction prior to KNN.  
- Multiple values of *k* are evaluated, and the best-performing configuration is selected using the validation set.


### Random Forest

- Random Forest is used as an ensemble-based model capable of capturing nonlinear feature interactions.  
- Class-weighted training is applied to address class imbalance, and the model is trained on a **15% stratified subset** of the training data for computational efficiency.



---

## Evaluation

Models are evaluated on the held-out validation set using:
- Accuracy
- ROC-AUC (primary metric)
- Confusion Matrix

Logistic Regression provides a strong and stable linear baseline.  
KNN trades some predictive performance for reduced computational cost.  
Random Forest demonstrates the strongest overall validation performance, particularly in terms of ROC-AUC.

---

## Confusion Matrix Interpretation

The confusion matrix for the Random Forest model indicates:
- A high number of true positive predictions, corresponding to correctly identified successful repayments
- Improved detection of default cases despite class imbalance
- A balanced trade-off between false positives and false negatives

Overall, this suggests that Random Forest provides reliable performance for risk-aware loan repayment prediction.

---

## Feature Importance Analysis (Random Forest)

Feature importance is analyzed using the trained Random Forest model, with one-hot encoded categorical features aggregated back to the original feature level.  

The analysis shows that **employment status**, **debt-to-income ratio**, and **credit score** are the most influential predictors of loan repayment.  
This provides additional interpretability for the Random Forest model and helps explain its strong validation performance.

---

## Model Selection and Test Set Inference

Based on validation performance, the Random Forest model is used to perform inference on the test set.  
The trained pipeline generates probability scores representing the likelihood that a loan will be paid back.  

Predicted probabilities are stored in a submission-style format with `id` and `loan_paid_back`, and the output distribution is inspected to ensure reasonable probability ranges.

---

## Repository Structure

```
Predicting-Loan-Payback/
├── Predicting_Loan_Payback.ipynb
├── train.csv
├── test.csv
├── sample_submission.csv
├── project_overview.pdf
└── README.md
```

---

## Conclusion

This project demonstrates a complete machine learning workflow for tabular classification, including:
- Robust preprocessing using pipelines
- Exploration of multiple baseline and ensemble models
- Consistent evaluation on a held-out validation set
- Explicit handling of class imbalance
- Generation of probability-based predictions

The use of `sklearn` pipelines ensures clean and reproducible experimentation, while comparing multiple models highlights important trade-offs between simplicity, efficiency, and predictive performance.  
Overall, the Random Forest model demonstrates the strongest validation performance among the evaluated methods, particularly in terms of ROC-AUC.
