# 🛡️ Customer Churn Prediction System

[![Python Version](https://img.shields.type/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.type/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.type/badge/xgboost-3.2-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit App](https://img.shields.type/badge/Streamlit-1.15+-ff4b4b.svg)](https://streamlit.io/)

A complete, portfolio-grade Machine Learning solution to predict customer attrition and recommend data-driven retention strategies. This project is designed to simulate a real-world enterprise data science workspace, featuring an automated data pipeline, exploratory analysis, hyperparameter tuning, model comparison, and an interactive Streamlit application.

---

## 📂 Project Structure

```
Customer-Churn-Prediction/
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv   # Automatically downloaded dataset
│   └── processed/
│       ├── train.csv                  # Split training data
│       └── test.csv                   # Split testing data
├── notebooks/
│   └── eda_and_modeling.ipynb         # EDA and training code
├── src/
│   ├── data_loader.py                 # Pulls CSV from raw source
│   ├── preprocessing.py               # Data cleaning & preprocessing
│   ├── train.py                       # Model tuning & Pipeline serialization
│   └── evaluate.py                    # Evaluates and creates charts
├── models/
│   ├── churn_pipeline.pkl             # Serialized inference Pipeline
│   └── training_summary.pkl           # Metric checkpoints
├── app/
│   ├── app.py                         # Streamlit application
│   ├── style.css                      # Custom theme overrides
│   └── static/                        # Shared evaluation plots
├── reports/
│   ├── figures/                       # Metrics plots (ROC, Confusion Matrix)
│   └── project_report.md              # Detailed final project report
├── requirements.txt                   # Dependency file
├── churn_prediction.ipynb             # Root Jupyter notebook
└── README.md                          # Repository documentation
```

---

## 📊 Model Comparison & Results

We trained and tuned **Logistic Regression**, **Random Forest**, and **XGBoost** using `GridSearchCV` on the F1-score to counter class imbalance.

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Best)** | **80.48%** | **65.52%** | **55.88%** | **60.32%** | **84.13%** |
| Random Forest | 80.20% | 66.55% | 51.07% | 57.79% | 83.90% |
| XGBoost | 80.77% | 67.58% | 52.94% | 59.37% | 84.45% |

### Selection Rationale:
**Logistic Regression** was chosen for final deployment due to having the highest test **F1-score (60.32%)** and higher recall, which ensures we catch a larger fraction of customers actually intending to churn. It also allows direct coefficient interpretability for CRM integrations.

---

## 💡 Key Business Insights & Playbook

Instead of focusing solely on metrics, this project includes a business playbook automatically loaded in the Streamlit application:

*   **Month-to-month contracts** are the single strongest churn driver (churn rate over 42%). Offering targeted discounts to transition them to 1-year contracts is a high-yield retention tactic.
*   **Electronic Check** payment users have high churn rates. Incentivizing automatic billing setup (credit card or bank transfer) with a ₹400 billing credit can reduce passive attrition.
*   Customers without **Online Security** and **Tech Support** show elevated churn risk. Proactive bundling of these services during onboarding increases retention sticky-ness.
*   High risk is concentrated during the **first 12 months** of customer tenure, signifying the need for structured onboarding checkpoints.

---

## 🚀 How to Run the Project

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Download and Train Pipeline
Run the data loader, preprocessing, training, and evaluation scripts:
```bash
# Download raw dataset
python src/data_loader.py

# Preprocess and split train/test sets
python src/preprocessing.py

# Train models, tune parameters, and export pipeline
python src/train.py

# Evaluate and export plots
python src/evaluate.py
```

### 3. Start Streamlit Application
Launch the web interface locally:
```bash
streamlit run app/app.py
```

