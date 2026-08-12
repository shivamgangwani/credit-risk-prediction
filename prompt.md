# AI Assistance Prompt

## Project Context

I am building my first machine learning project as part of my preparation for quantitative and data-oriented roles.

My background is in economics and statistics, and I am currently studying quantitative economics. I have a reasonably strong foundation in statistics, econometrics, mathematics, and Python, but my understanding of machine learning is still relatively rudimentary.

The project uses a public credit-risk dataset to predict whether a borrower will experience serious delinquency within two years. The goal is not to achieve the highest possible predictive accuracy or use sophisticated algorithms. The goal is to demonstrate that I can:

- formulate a real business prediction problem;
- perform exploratory data analysis;
- prepare data appropriately;
- build and compare statistical and machine-learning models;
- evaluate models using appropriate metrics;
- interpret model results;
- connect modelling decisions to the underlying business problem; and
- communicate the analysis clearly.

The project is intended to be published on GitHub and discussed in interviews.

## Modelling Approach

The project compares several classification models, including:

- Logistic Regression
- Linear Discriminant Analysis
- Quadratic Discriminant Analysis
- Decision Tree
- Random Forest

The analysis focuses on ROC-AUC, Average Precision, precision, recall, F1 score, cross-validation, model interpretation, and classification thresholds.

The project places particular emphasis on understanding the trade-offs between model performance, interpretability, and practical use.

## How I Want AI to Help

Help me understand and improve the project rather than simply producing work for me.

When reviewing the analysis:

- Check whether statistical and machine-learning interpretations are technically correct.
- Point out errors, questionable assumptions, or unsupported conclusions.
- Explain concepts when I do not understand them.
- Prefer simple explanations over unnecessary jargon.
- Distinguish clearly between prediction and causal inference.
- Do not recommend more sophisticated methods simply because they exist.
- Keep the scope appropriate for a first ML project.

When reviewing code, focus on correctness, clarity, and good practice. Do not unnecessarily rewrite code that is already correct.

When reviewing the written material, focus particularly on:

- clarity;
- concision;
- readability;
- technical accuracy;
- avoiding unnecessary ML jargon;
- avoiding repetition; and
- making sure interpretations do not overstate what the analysis shows.

The writing should sound like a quantitative economics student explaining their own analysis, not like an ML textbook or a professional data-science consultancy report.

## Writing Style

The project should use plain, direct language.

Prefer:

> "The model predicts which borrowers are at higher risk."

over:

> "The model facilitates the identification of borrowers exhibiting elevated predicted risk."

Keep technical terminology when it is necessary to describe the analysis, but remove jargon that does not add information.

Avoid lengthy explanations of standard concepts when the notebook already demonstrates them through code, tables, or figures.

For markdown cells, generally use:

**question → result → brief interpretation**

rather than long methodological explanations.

The notebooks should explain the reasoning behind important decisions, but should not attempt to teach the reader all of machine learning.

## Business Interpretation

The project should connect statistical results to the practical credit-risk problem.

In particular, explain the distinction between:

- ranking borrowers by predicted risk; and
- deciding which borrowers to actually flag.

For example, a lower classification threshold can increase recall but also generate more false positives. Flagging every borrower would achieve 100% recall, but precision would fall to the underlying delinquency rate, making the resulting flags impractical if they require manual review.

Similarly, ROC-AUC should be interpreted as a measure of how well the model ranks borrowers across thresholds, rather than as a direct measure of how useful a particular classification rule will be in practice.

Where appropriate, discuss how false positives, false negatives, review capacity, and economic costs would affect a real-world threshold choice.

## General Principle

The project should demonstrate statistical thinking and quantitative modelling rather than algorithmic sophistication.

Do not make the project sound more advanced than it is. The objective is to show that I understand the methods I have used, can evaluate them properly, and can communicate their results clearly.