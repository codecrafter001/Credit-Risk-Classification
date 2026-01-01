# Credit-Risk-Classification
This project builds a machine learning system to assess loan applicant risk using structured financial data.  
It demonstrates how data-driven models can support credit decisioning in real-world FinTech scenarios.

---

## Objective

Predict whether a loan applicant is:
- **Low Risk (0)**
- **High Risk (1)**

Based on key financial and behavioral indicators.

---

## Dataset

Each dataset follows the same structure but contains different values to simulate real-world data variation.

### Features
- Monthly Income  
- Credit Score  
- Existing EMIs  
- Employment Type  
- Past Defaults  

**Target:** `risk_label`

---

## Approach

The project implements two models:
- **Logistic Regression** – interpretable baseline for credit risk modeling
- **Random Forest** – non-linear model for improved performance and feature importance

---

## Evaluation

Model performance is evaluated using:
- Accuracy  
- Precision  
- Recall  
- Confusion Matrix  

These metrics reflect common trade-offs in financial risk assessment.

---

## Key Question

**Which features contribute most to high-risk classification, and why?**

Participants are expected to justify their findings using model outputs and feature importance.

---

## How to Run (Google Colab)

```python
!git clone https://github.com/<YOUR-USERNAME>/credit-risk-classification-ml.git
%cd credit-risk-classification-ml
````

Select a dataset inside the notebook:

```python
DATASET = "datasets/credit_risk_groupA.csv"
```

---





Just say **next** 🚀
```
