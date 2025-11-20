# Random Forest – LendingClub Loan Repayment

For this project we explore publicly available data from **LendingClub.com**.  
LendingClub connects people who need money (**borrowers**) with people who have money (**investors**).

As an investor, you’d want to invest in borrowers who are most likely to pay you back in full.  
In this project, we use a **Random Forest classifier** to predict whether a borrower will pay back their loan in full based on their profile and loan information.

We use lending data from **2007–2010** and try to classify and predict whether or not the borrower paid back their loan in full (via the `not.fully.paid` target column).

---

## 🔧 What the script does

`loan_random_forest.py`:

1. **Loads** the loan dataset from `loan_data.csv`
2. **Explores** the data by generating:
   - A pairplot of selected numeric features plus the target (`not.fully.paid`)
   - A correlation heatmap for numeric columns
3. **Preprocesses** the data:
   - One-hot encodes the categorical `purpose` column
   - Keeps all numeric columns as features
   - Defines the target:
     - `not.fully.paid` – 1 if the borrower did **not** pay back the loan in full, 0 otherwise
4. **Splits** the data into train and test sets (stratified by the target)
5. **Trains** a `RandomForestClassifier` on the training data
6. **Evaluates** the model on the test data using:
   - Accuracy
   - Confusion matrix
   - Classification report (precision, recall, F1-score)
   - ROC curve and AUC
7. **Analyzes feature importance**:
   - Extracts `feature_importances_` from the Random Forest
   - Saves them as plots and tables
8. **Saves all results** into a `results/` directory:
   - Plots as `.png`
   - Metrics and importances as `.txt` / `.csv`

---

## 📚 Dataset description (LendingClub loans)

We use a LendingClub loan dataset covering **2007–2010**.  
Each row represents a loan, with columns like:

- `credit.policy`:  
  1 if the customer meets the credit underwriting criteria of LendingClub.com, 0 otherwise.

- `purpose`:  
  The purpose of the loan. Possible values include:  
  `"credit_card"`, `"debt_consolidation"`, `"educational"`, `"major_purchase"`, `"small_business"`, and `"all_other"`.

- `int.rate`:  
  The interest rate of the loan, as a proportion (a rate of 11% is stored as `0.11`).  
  Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.

- `installment`:  
  The monthly installments owed by the borrower if the loan is funded.

- `log.annual.inc`:  
  The natural log of the self-reported annual income of the borrower.

- `dti`:  
  The debt-to-income ratio of the borrower (amount of debt divided by annual income).

- `fico`:  
  The borrower’s FICO credit score.

- `days.with.cr.line`:  
  The number of days the borrower has had a credit line.

- `revol.bal`:  
  The borrower’s revolving balance (amount unpaid at the end of the credit card billing cycle).

- `revol.util`:  
  The borrower’s revolving line utilization rate (amount of the credit line used relative to total credit available).

- `inq.last.6mths`:  
  The number of inquiries by creditors in the last 6 months.

- `delinq.2yrs`:  
  The number of times the borrower had been 30+ days past due on a payment in the past 2 years.

- `pub.rec`:  
  The number of derogatory public records (bankruptcy filings, tax liens, or judgments).

- `not.fully.paid`:  
  Target label – 1 if the borrower did **not** pay back the loan in full, 0 if they did.

The main task is to **predict `not.fully.paid`** from the other features.

---

## 📁 Outputs (in `results/`)

After running the script, you’ll get:

**Exploratory plots**

- `loan_pairplot.png`  
  Pairplot of selected numeric features (e.g. `fico`, `int.rate`, `installment`, `log.annual.inc`, `dti`, `credit.policy`) colored by `not.fully.paid`.

- `loan_corr_heatmap.png`  
  Correlation matrix heatmap showing relationships between numeric features and the target.

**Model performance**

- `loan_random_forest_confusion_matrix.png`  
  Confusion matrix heatmap for the Random Forest model.

- `loan_random_forest_roc_curve.png`  
  ROC curve with AUC value in the legend.

- `loan_random_forest_report.txt`  
  Accuracy, confusion matrix, and full classification report.

**Feature importance**

- `loan_random_forest_feature_importances.png`  
  Horizontal bar chart of feature importances from the Random Forest.

- `loan_random_forest_feature_importances.csv`  
  Table of features and their importance values (CSV).

- `loan_random_forest_feature_importances.txt`  
  Human-readable list of feature importances, one per line.

These results can be viewed directly on GitHub so anyone can inspect the model’s performance and which features matter most, without having to run the code.
