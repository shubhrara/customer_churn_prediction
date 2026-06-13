# Final-Year Project Report: Customer Churn Prediction and Retention Portal

**Project Title:** Customer Churn Prediction System using Advanced Machine Learning  
**Domain:** Data Science & Predictive Analytics  
**Academic Year:** 2025 - 2026  

---

## Abstract
Customer retention is a vital driver of profitability for subscription-based telecommunication providers. Retaining an existing customer is estimated to be 5 to 25 times cheaper than acquiring a new one. This project presents a complete, automated end-to-end Machine Learning solution built on the **IBM Telco Customer Churn** dataset. Using Python, Scikit-learn, and XGBoost, we implement rigorous cleaning, Exploratory Data Analysis (EDA), feature scaling, and categorical encoding. Three candidate models (Logistic Regression, Random Forest, and XGBoost) are optimized using 5-fold cross-validated Grid Search. Logistic Regression emerged as the optimal classifier for deployment, balancing execution speed, model interpretability, and class-balanced F1-score (60.3% test F1-score, 80.5% test accuracy, and 84.1% ROC-AUC). We export this as a unified Scikit-learn Pipeline and deploy it through an interactive, glassmorphic Streamlit application equipped with a real-time risk classification and customized customer retention recommendation playbook.

---

## 1. Introduction & Problem Statement
Customer attrition (or churn) refers to the loss of clients or subscribers. In telecom industries, where product offerings are highly competitive, customers can easily switch providers. 

The primary business goal is to:
1. **Identify** high-risk customers before they churn.
2. **Understand** the root causes of their dissatisfaction (feature analysis).
3. **Recommend** specific, tailored incentives to retain them.

This project delivers a machine learning solution that fits into existing CRM software by outputting both a churn classification label (Churn / No Churn) and the associated probability score, coupled with automated business actions.

---

## 2. Exploratory Data Analysis (EDA) & Key Findings
Through extensive exploratory analysis of the 7,043 customers, several structural attrition drivers were identified:
*   **Onboarding & Tenure Risk:** A steep distribution of churn was observed in the first 12 months. Retention efforts are critical during onboarding.
*   **Contractual Vulnerability:** Month-to-month contracts exhibit an alarming churn rate exceeding **42.7%**, compared to only 11.2% for 1-year and 2.8% for 2-year contracts.
*   **Payment Friction:** Customers utilizing Electronic Checks churn at **45.2%**, signaling transaction friction or billing dissatisfaction. Maaile Checks, credit cards, and bank transfers remain stable (under 20% churn).
*   **Lack of Value Add-ons:** Customers who do not subscribe to security features like "Online Security" or "Tech Support" churn at rates twice as high as those who do.

---

## 3. Methodology & System Architecture
The application pipeline follows a modern, modular design:

```
[Raw CSV Dataset]
       │
       ▼
[Data Cleaning] (Handle spaces in TotalCharges, Map Churn to 0/1, Drop IDs)
       │
       ▼
[Train/Test Split] (80/20 Stratified Split)
       │
       ▼
[Preprocessing Pipeline] (StandardScaler for Numerical, OneHotEncoder for Categorical)
       │
       ▼
[Hyperparameter Tuning] (5-Fold Grid Search over LogisticRegression, RandomForest, XGBoost)
       │
       ▼
[Model Comparison & Selection] (Select model maximizing F1-Score)
       │
       ▼
[Unified Pipeline Export] (Save Pipeline(preprocessor, model) as churn_pipeline.pkl)
       │
       ▼
[Streamlit App Interface] (Load pickle, input customer features, display prediction & tips)
```

### Preprocessing Specifications
*   **Numerical Features (`tenure`, `MonthlyCharges`, `TotalCharges`):** Standardized using Z-score scaling to bring values to a comparable mean-zero scale.
*   **Categorical Features (16 variables):** Encoded using One-Hot encoding. To prevent columns alignment mismatches during production inference, the categorical encoder is bundled directly inside the serialized pipeline.

---

## 4. Model Training & Comparison
We trained and evaluated three distinct algorithms:
1.  **Logistic Regression (Baseline & Explainer):** Fast, highly interpretable, optimized with L1/L2 penalties.
2.  **Random Forest (Ensemble Bagging):** Non-linear trees, robust to outliers, tuned for tree depth and estimators.
3.  **XGBoost (Extreme Gradient Boosting):** Advanced boosting framework using regularization to prevent overfitting.

### Hyperparameter Tuning Grids
*   **Logistic Regression:** `C`: `[0.01, 0.1, 1.0, 10.0]`, `solver`: `['liblinear', 'lbfgs']`
*   **Random Forest:** `n_estimators`: `[100, 200]`, `max_depth`: `[5, 10, None]`
*   **XGBoost:** `learning_rate`: `[0.01, 0.1, 0.2]`, `max_depth`: `[3, 5, 7]`, `n_estimators`: `[100, 200]`

---

## 5. Experimental Results
All models were evaluated on an independent, stratified 20% test dataset (1,409 customers). F1-score was selected as the optimization metric due to the class imbalance (26.5% churned class).

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Best)** | **80.48%** | **65.52%** | **55.88%** | **60.32%** | **84.13%** |
| Random Forest | 80.20% | 66.55% | 51.07% | 57.79% | 83.90% |
| XGBoost | 80.77% | 67.58% | 52.94% | 59.37% | 84.45% |

### Selection Rationale
While XGBoost achieved slightly higher test accuracy (80.77% vs 80.48%) and ROC-AUC (84.45% vs 84.13%), **Logistic Regression** was selected as the production model. It achieved the highest test F1-score (**60.32%**) and a higher recall (55.88% vs 52.94%), meaning it captures more churners. Furthermore, Logistic Regression allows for transparent coefficient-based explaining, which is critical for compliance and trust in business decisions.

---

## 6. Business Insights & Actionable Playbook
Based on the absolute values of the coefficients of the Logistic Regression model, the following business actions are proposed:

1.  **Contract Mitigation:** Establish a marketing campaign targeting Month-to-Month accounts. Convert them to 1-Year plans by offering a ₹800/month contract discount. This changes their risk footprint instantly.
2.  **Payment Friction Reduction:** Offer a one-time ₹400 bill credit to customers paying via Electronic Check who agree to set up Automatic Credit Card or Bank Debit autopayments.
3.  **Cross-selling Core Protections:** Target customers with Internet Service who did *not* opt for Online Security or Tech Support. Bundle these security packages at a discounted rate (e.g., ₹250/month add-on) to increase product attachment.
4.  **Fiber Optic Product Check:** Customers on Fiber Optic show unexpectedly high churn rates. Customer success agents should follow up within 30 days of installation to resolve speed and routing quality feedback.

---

## 7. Streamlit Web App Interface
The Streamlit application features a premium dark theme and is divided into:
1.  **Interactive Predictor:** Includes sliders and dropdowns corresponding to customer demographics, payment details, and services. Displays class probability with a dynamic progress bar.
2.  **Automated Playbook:** Displays customized recommendation cards based on the customer's attributes (e.g., advising payment updates if they pay by check, or tech support trials if they lack security features).
3.  **Analytics Page:** Embeds visual graphs of ROC curves, confusion matrices, and feature importances for stakeholder transparency.

---

## 8. Portfolio Deliverables

### Resume-Ready Project Description
*   **Customer Churn Prediction System | Python, Scikit-learn, XGBoost, Streamlit, Pandas**
    *   Designed and built an end-to-end machine learning pipeline to predict customer churn, achieving **80.5% accuracy** and **0.841 ROC-AUC** on a dataset of 7,040+ telecom customers.
    *   Optimized Logistic Regression, Random Forest, and XGBoost models using Grid Search CV; encapsulated preprocessing (OneHotEncoder, StandardScaler) and estimator into a unified Scikit-learn Pipeline.
    *   Developed and deployed an interactive Streamlit web application showcasing real-time inference, predictive probabilities, and an automated customer retention playbook generating custom business recommendations.

### GitHub Project Description
> 🚀 Complete final-year machine learning project predicting customer attrition with 80.5% accuracy. Built with Python, Scikit-Learn, and XGBoost. Features a unified pipeline export, comprehensive Jupyter notebooks, feature importance analysis, business recommendations, and an interactive Streamlit application dashboard.
