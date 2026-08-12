# Credit Risk Modelling

## Executive Summary

This project uses the **Give Me Some Credit** dataset to predict whether a borrower will experience serious delinquency within two years. I compare Logistic Regression, LDA, QDA, Decision Trees, and Random Forests, focusing on predictive performance, interpretability, and classification thresholds.

- **150,000 borrowers**, of whom 6.68% experienced serious delinquency.
- **LDA** achieved the highest test ROC-AUC (**0.851**), followed by Random Forest (**0.849**) and Logistic Regression (**0.839**).
- Previous delinquency history was particularly informative in Logistic Regression, while credit utilisation, debt ratio, income, and age were most important in the Random Forest.
- Lowering the classification threshold substantially improves recall, but also increases false positives.

## 1. Business Problem

Credit providers need to identify borrowers who are at higher risk of serious delinquency.

This project predicts whether a borrower will experience serious delinquency within the following two years, based on their financial and credit characteristics at the time of prediction.

A useful model could support credit-risk screening and prioritisation.

## 2. Data

The project uses the **Give Me Some Credit** dataset. It contains 150,000 borrowers and 10 predictor variables covering:

- Age and dependents
- Income
- Debt ratio
- Revolving credit utilisation
- Credit lines and loans
- Previous delinquency history

The positive class is relatively rare: **6.68%** of borrowers experienced serious delinquency. Because of this class imbalance, accuracy is not used as the primary metric. The analysis focuses on ROC-AUC, Average Precision, precision, recall, and F1.

## 3. Data Preparation

The EDA identified missing values, an invalid `age = 0` observation, and `98` as a special missing-value code in the delinquency variables.

Missing values are median-imputed, highly skewed predictors are transformed using $\log(1+x)$, and predictors are standardised where appropriate. All preprocessing is performed within sklearn pipelines to prevent data leakage.

## 4. Modelling Approach

I compare six models:

1. Dummy Classifier
2. Logistic Regression
3. Linear Discriminant Analysis
4. Quadratic Discriminant Analysis
5. Decision Tree
6. Random Forest

Models are evaluated using a stratified 80/20 train-test split and 5-fold stratified cross-validation. Performance is assessed using ROC-AUC, Average Precision, precision, recall, and F1.

The analysis also considers model interpretability and the choice of classification threshold.

## 5. Model Performance

### Held-out test set

| Model | ROC-AUC | Average Precision | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| **LDA** | **0.851** | **0.362** | 0.446 | 0.324 | 0.375 |
| Random Forest | 0.849 | 0.356 | 0.523 | 0.179 | 0.266 |
| Logistic Regression | 0.839 | 0.358 | 0.542 | 0.163 | 0.250 |
| QDA | 0.830 | 0.344 | 0.326 | 0.529 | 0.404 |
| Decision Tree | 0.610 | 0.119 | 0.256 | 0.275 | 0.265 |
| Dummy Classifier | 0.500 | 0.067 | 0.000 | 0.000 | 0.000 |

LDA performs best on ROC-AUC and Average Precision, although the differences between LDA, Random Forest, and Logistic Regression are small. The Decision Tree performs substantially worse than the other models.

### Cross-validation

| Model | Mean ROC-AUC | SD |
|---|---:|---:|
| **LDA** | **0.846** | 0.007 |
| Random Forest | 0.840 | 0.004 |
| Logistic Regression | 0.833 | 0.007 |
| QDA | 0.829 | 0.005 |
| Decision Tree | 0.610 | 0.007 |
| Dummy Classifier | 0.500 | 0.000 |

The cross-validation results produce essentially the same model ranking as the held-out test set.

## 6. What Drives Credit Risk?

### Logistic Regression

The strongest positive associations are concentrated in **previous delinquency behaviour**.

Holding other predictors fixed, a one-standard-deviation increase in the transformed delinquency variables is associated with substantially higher odds of future serious delinquency:

- 30–59 days past due: approximately **51% higher odds**
- 90+ days late: approximately **49% higher odds**
- 60–89 days past due: approximately **27% higher odds**
- Revolving credit utilisation: approximately **22% higher odds**

Age and monthly income have the strongest negative associations.

Overall, **past repayment behaviour contains substantial information about future credit risk**.

### Random Forest

The most important predictors by feature importance are:

- Revolving credit utilisation: **19.5%**
- Debt ratio: **18.0%**
- Monthly income: **14.8%**
- Age: **12.9%**

Together, these four variables account for approximately two-thirds of total feature importance.

The rankings differ between the two models, showing that models can use the same predictors in different ways. Feature importance should be interpreted as predictive, not causal.

## 7. Classification Thresholds

The model produces a probability of delinquency, but a real application must decide which borrowers to flag.

A lower threshold catches more delinquent borrowers but also creates more false positives. If every borrower is flagged, recall reaches 100%, but precision falls to the delinquency rate. Most flags would therefore require review without identifying a delinquent borrower.

Thresholds are therefore a business decision as well as a statistical one. In this analysis, thresholds are selected by maximising F1 using out-of-fold training predictions and then applied to the held-out test set.

| Model | Selected Threshold | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.14 | 0.380 | 0.465 | 0.418 |
| LDA | 0.12 | 0.369 | 0.485 | 0.419 |
| QDA | 0.50 | 0.326 | 0.529 | 0.404 |
| Random Forest | 0.26 | 0.391 | 0.458 | **0.422** |

In practice, the threshold should ultimately be chosen using the economic costs of false positives and false negatives and the capacity available for reviewing flagged borrowers.

## 8. Limitations

- **No temporal validation:** Random train-test splitting may overstate performance on future borrowers.
- **Target definition:** The target captures serious delinquency within two years but not the full financial cost of credit risk.
- **Threshold selection:** F1 does not incorporate the economic costs of different classification errors.
- **Probability calibration:** The analysis focuses on discrimination rather than whether predicted probabilities are well calibrated.
- **Observational data:** The models identify predictive relationships, not causal effects.

## 9. Repository Structure

```text
.
├── README.md
├── data/
│   └── ...
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
└── requirements.txt
```

**Notebooks**
* **[`01_eda.ipynb`](notebooks/01_eda.ipynb)** — Exploratory analysis, data-quality investigation, transformations, and preprocessing decisions.
* **[`02_modeling.ipynb`](notebooks/02_modeling.ipynb)** — Model estimation, cross-validation, performance evaluation, interpretation, and threshold analysis.


**Tools**
Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, Jupyter Notebook.



## 10. AI Usage
AI tools were used during the development of this project, primarily for reviewing code, debugging, discussing modelling choices, and editing documentation for clarity.

The final analysis, modelling decisions, interpretations, and conclusions were reviewed and validated by me.

The prompts used during development are included in [`prompt.md`](prompt.md) for transparency.