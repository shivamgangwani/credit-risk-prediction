# Credit Risk Modelling

## Executive Summary

This project develops and evaluates statistical and machine learning models for predicting whether a borrower will experience **serious delinquency within the next two years**.

Using the public **Give Me Some Credit** dataset, I formulate the problem as a borrower-level binary classification task and compare Logistic Regression, Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), Decision Trees, and Random Forests.

The main findings are:

* The dataset contains **150,000 borrowers**, of whom approximately **6.68%** experienced serious delinquency within two years.
* **LDA achieved the highest ROC-AUC (0.851)** on the held-out test set, followed closely by Random Forest (0.849) and Logistic Regression (0.839).
* Cross-validation produces a similar ranking, with LDA achieving mean ROC-AUC of **0.846** and Random Forest **0.840**.
* Logistic Regression remains competitive despite being substantially simpler and more interpretable.
* Previous delinquency history is particularly informative in the logistic model, while measures of credit utilisation and indebtedness receive greater importance in the Random Forest.
* Selecting a classification threshold using out-of-fold predictions substantially improves recall relative to the default 0.5 threshold, illustrating the importance of separating **risk ranking** from **decision thresholds**.

The results suggest that relatively standard statistical models can provide meaningful predictive performance for this problem, without requiring highly complex machine learning methods.

---

## 1. Business Problem

Credit providers need to distinguish between borrowers who are likely to remain current on their obligations and those who may become seriously delinquent.

The objective of this project is to estimate:

$P(Y_i = 1 \mid X_i)$

where:

* $(Y_i = 1)$ if borrower ($i$) experiences serious delinquency within the following two years;
* $(Y_i = 0)$ otherwise;
* $(X_i)$ represents the borrower's financial and credit characteristics observed at the time of prediction.

This is a **cross-sectional binary classification problem at the borrower level**.

A useful predictive model could support credit-risk screening and prioritisation by identifying borrowers with elevated predicted risk.

---

## 2. Data

The analysis uses the [Give Me Some Credit dataset](https://www.kaggle.com/c/GiveMeSomeCredit) originally released through Kaggle.

The dataset contains 150,000 borrower observations and 10 predictor variables covering:

* Age and number of dependents
* Monthly income
* Debt ratio
* Revolving credit utilisation
* Number of open credit lines and loans
* Number of real-estate loans
* Previous delinquency history

The target variable, `SeriousDlqin2yrs`, indicates whether the borrower experienced serious delinquency within two years.

### Class imbalance

Only **6.68%** of observations belong to the positive class.

This makes accuracy an inadequate standalone measure of performance. The analysis therefore focuses primarily on **ROC-AUC, Average Precision, precision, recall, and F1 score**.

---

## 3. Data Preparation

The exploratory analysis identified several data-quality and distributional issues.

### Missing values

`MonthlyIncome` has approximately **19.8% missing observations**, while `NumberOfDependents` has approximately **2.6%** missing observations.

Missing values are handled using **median imputation within the modelling pipeline**, ensuring that imputation parameters are estimated only from the relevant training data.

### Special values

The delinquency variables contain the value `98` in exactly the same 264 observations. This value is far outside their normal distributions and appears to represent a special missing-value code rather than a genuine delinquency count.

These values are therefore treated as missing.

### Invalid age

One observation has `age = 0`. This is treated as an invalid observation and removed from the modelling sample.

### Transformations

Several highly right-skewed predictors are transformed using:

$\log(1+x)$

The transformed variables are then standardised where appropriate.

All preprocessing operations are incorporated into sklearn pipelines, preventing information from the test set from entering the training process.

---

## 4. Modelling Approach

Six models are evaluated:

1. Dummy Classifier
2. Logistic Regression
3. Linear Discriminant Analysis
4. Quadratic Discriminant Analysis
5. Decision Tree
6. Random Forest

A **stratified 80/20 train-test split** is used, preserving the class distribution.

Model selection and comparison additionally use **5-fold stratified cross-validation** on the training data.

The modelling objective is not simply to maximise predictive accuracy. Instead, the analysis examines the trade-offs between:

* Predictive discrimination
* Precision and recall
* Model complexity
* Interpretability
* Classification thresholds

---

## 5. Model Performance

### Held-out test set

| Model               |   ROC-AUC | Average Precision | Precision | Recall |    F1 |
| ------------------- | --------: | ----------------: | --------: | -----: | ----: |
| **LDA**             | **0.851** |         **0.362** |     0.446 |  0.324 | 0.375 |
| Random Forest       |     0.849 |             0.356 |     0.523 |  0.179 | 0.266 |
| Logistic Regression |     0.839 |             0.358 |     0.542 |  0.163 | 0.250 |
| QDA                 |     0.830 |             0.344 |     0.326 |  0.529 | 0.404 |
| Decision Tree       |     0.610 |             0.119 |     0.256 |  0.275 | 0.265 |
| Dummy Classifier    |     0.500 |             0.067 |     0.000 |  0.000 | 0.000 |

The **Dummy Classifier** provides a useful baseline: its ROC-AUC is 0.5 and its Average Precision is approximately equal to the positive-class prevalence.

LDA achieves the highest ROC-AUC and Average Precision, although the differences between LDA, Logistic Regression, and Random Forest are relatively small.

The Decision Tree performs substantially worse than the other models, suggesting that a single tree is insufficient to capture the structure of the prediction problem.

### Cross-validation

| Model               | Mean ROC-AUC |    SD |
| ------------------- | -----------: | ----: |
| **LDA**             |    **0.846** | 0.007 |
| Random Forest       |        0.840 | 0.004 |
| Logistic Regression |        0.833 | 0.007 |
| QDA                 |        0.829 | 0.005 |
| Decision Tree       |        0.610 | 0.007 |
| Dummy Classifier    |        0.500 | 0.000 |

The cross-validation results produce essentially the same model ranking, providing additional evidence that the performance differences are not solely an artefact of the particular test split.

---

## 6. What Drives Credit Risk?

The models provide complementary perspectives on the predictors of serious delinquency.

### Logistic Regression

The strongest positive associations are concentrated in **previous delinquency behaviour**.

Holding other predictors fixed, a one-standard-deviation increase in the transformed delinquency variables is associated with substantially higher odds of future serious delinquency.

For example:

* 30–59 days past due: approximately **51% higher odds**
* 90+ days late: approximately **49% higher odds**
* 60–89 days past due: approximately **27% higher odds**
* Revolving credit utilisation: approximately **22% higher odds**

Age and monthly income have the largest negative coefficients, with higher values associated with lower predicted odds of serious delinquency.

This provides a relatively interpretable picture: **past repayment behaviour contains substantial information about future credit risk**.

### Random Forest

The Random Forest produces a somewhat different importance ranking.

The most important predictors by impurity-based feature importance are:

* Revolving credit utilisation: **19.5%**
* Debt ratio: **18.0%**
* Monthly income: **14.8%**
* Age: **12.9%**

Together, these four variables account for approximately two-thirds of total feature importance.

The difference between the Logistic Regression and Random Forest rankings illustrates an important modelling point: different algorithms can exploit different aspects of the same predictors.

Feature importance should therefore be interpreted as **model-specific predictive importance**, rather than as evidence of causal effects.

---

## 7. Classification Thresholds

A predicted probability is not itself a classification decision. A threshold must be chosen to determine which borrowers are classified as high risk.

The default threshold of 0.5 produces relatively low recall for several models.

Instead, thresholds are selected by maximising F1 using **out-of-fold predictions generated exclusively from the training data**. The selected threshold is then locked and applied once to the untouched test set.

| Model               | Selected Threshold | Test Precision | Test Recall |   Test F1 |
| ------------------- | -----------------: | -------------: | ----------: | --------: |
| Logistic Regression |               0.14 |          0.380 |       0.465 |     0.418 |
| LDA                 |               0.12 |          0.369 |       0.485 |     0.419 |
| QDA                 |               0.50 |          0.326 |       0.529 |     0.404 |
| Random Forest       |               0.26 |          0.391 |       0.458 | **0.422** |

The results demonstrate the fundamental **precision-recall trade-off**: increasing recall requires accepting more false positives.

In a real credit-risk application, the threshold should ultimately be determined using an explicit economic cost function rather than F1 alone.

---

## 8. Key Business Takeaways

### 1. Previous repayment behaviour is highly predictive

Borrowers with a history of serious delinquency are substantially more likely to experience future delinquency.

### 2. Model complexity does not automatically improve performance

Random Forest performs very well, but its ROC-AUC is only marginally higher than Logistic Regression and slightly below LDA.

This suggests that additional model complexity provides limited incremental discrimination for this dataset.

### 3. Ranking and classification are different problems

A model can be effective at ranking borrowers by risk without producing useful classifications at the default 0.5 threshold.

The threshold analysis demonstrates why operational credit decisions require a threshold aligned with the costs of false positives and false negatives.

### 4. Interpretability remains valuable

Given the relatively small performance gap between the models, Logistic Regression provides a compelling benchmark because its coefficients offer a transparent interpretation of the predictors associated with risk.

---

## 9. Limitations

* **No temporal validation:** The analysis uses a random train-test split rather than a time-based validation scheme. Performance may therefore differ when applied to future borrowers under changing economic conditions.
* **Target definition:** The target captures serious delinquency within two years but does not capture the full range of financial outcomes or costs associated with credit risk.
* **Class imbalance:** Only 6.68% of observations are positive, making accuracy a poor standalone measure of performance.
* **Threshold selection:** F1 provides a symmetric precision-recall criterion but does not explicitly incorporate the economic costs of different classification errors.
* **Probability calibration:** The analysis focuses primarily on discrimination rather than calibration. High ROC-AUC does not imply that predicted probabilities are well calibrated.
* **Observational data:** The models identify predictive relationships, not causal effects. Feature importance should therefore not be interpreted as evidence that changing a predictor would causally change delinquency risk.

---

## 10. Repository Structure

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

### Notebooks

* **[`01_eda.ipynb`](notebooks/01_eda.ipynb)** — Exploratory analysis, data-quality investigation, transformations, and preprocessing decisions.
* **[`02_modeling.ipynb`](notebooks/02_modeling.ipynb)** — Model estimation, cross-validation, performance evaluation, interpretation, and threshold analysis.

---

## Technical Stack

* Python
* pandas
* NumPy
* matplotlib
* seaborn
* scikit-learn
* Jupyter Notebook

---

## Conclusion

The analysis shows that serious delinquency can be predicted with meaningful discrimination using a relatively small set of borrower financial and credit characteristics.

LDA achieves the strongest overall discrimination, while Random Forest and Logistic Regression remain competitive. More importantly, the similarity in performance across several models demonstrates that **careful data preparation, appropriate evaluation, and thoughtful threshold selection can matter as much as algorithmic complexity**.

For practical credit-risk deployment, the next steps would be to evaluate temporal stability, probability calibration, and an economically motivated cost function for determining the operating threshold.
