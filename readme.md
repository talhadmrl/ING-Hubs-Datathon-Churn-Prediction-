# Customer Churn Prediction Project

This project aims to build an end-to-end machine learning model to predict customers' future **churn** risk (Customer Attrition) using customer demographics and transaction history data. This is a classification problem, and a highly optimized **XGBoost (Extreme Gradient Boosting)** model is utilized on a feature-rich dataset created through extensive feature engineering.

## 1. Datasets and Exploratory Data Analysis (EDA)

The project utilizes four main datasets:
* `customers.csv`: Contains demographic and general information for customers.
* `customer_history.csv`: Contains customers' monthly transaction histories and financial activity data.
* `reference_data.csv`: The reference data for the training set, including customer IDs and the target variable (`churn` - Churn: 1, Retention: 0).
* `reference_data_test.csv`: Contains customer IDs for the final test set.

### Data Dimensions:
* `customers`: 176,293 rows, 8 columns
* `customer_history`: 5,359,609 rows, 7 columns
* `reference_data (train)`: 133,287 rows, 3 columns
* `reference_data_test`: 43,006 rows, 2 columns

### Target Variable Distribution:
The **Churn Rate** in the training data (`reference_data`) is found to be **14.16%**. This indicates a significant **class imbalance**. To address this, the `scale_pos_weight` for the positive class (churn) is calculated as **6.06** (Retained / Churned) for the XGBoost model.

## 2. Feature Engineering

Extensive features are engineered from the `customer_history` table to capture not only the customer's current state but also behavioral changes over time.

### 2.1 Transaction History Aggregation
The following statistical summaries are calculated for each customer using all numerical columns in `customer_history` (EFT amounts, credit card transactions, etc.):
* Mean, Standard Deviation, Maximum, Minimum, Sum, and Median.

### 2.2 Behavioral Trend Analysis (Last 3 Months vs. First 3 Months)
* The average transaction values for each customer's **last 3 months** and **first 3 months** are calculated.
* The difference between these averages (`last3m_mean - first3m_mean`) is derived as a **trend** feature to indicate increasing or decreasing activity.

### 2.3 Activity Summaries
* `transaction_count`: The total number of transactions.
* `active_months`: The count of unique months the customer was active.

**Result:** A total of **62 rich features** were created after merging with demographic data.

## 3. Data Preprocessing

A consistent preprocessing pipeline is applied to both the train and test data:

1.  **Missing Value Imputation (Transaction Data):** Missing transaction counts/amounts are filled with **0** to represent "no transaction".
2.  **Categorical Data Handling:** Missing `work_sector` values are labeled as 'Missing', and **One-Hot Encoding** is applied to all categorical columns.
3.  **Remaining Missing Value Imputation:** Any remaining NaN values in numerical columns (e.g., in `std` or `trend` features) are filled with the column's **median**.
4.  **Data Alignment:** Training and test sets are aligned to ensure they have the exact same set of feature columns.

## 4. Modeling: XGBoost

**XGBoost (Extreme Gradient Boosting)** classifier is used for prediction.

### 4.1 Model Parameters and Optimization
The model is optimized to handle class imbalance and mitigate overfitting:
* `objective`: 'binary:logistic'
* `eval_metric`: 'auc'
* `n_estimators`: 1000
* `learning_rate`: 0.03 (Lower rate for better generalization)
* **`scale_pos_weight`**: 6.06 (Handles class imbalance)
* `early_stopping_rounds`: 50 (Prevents overfitting)
* `subsample` and `colsample_bytree`: 0.7 (Further reduces variance)



## 6. Feature Importance

The top features highlight the importance of behavioral data:

| Feature | Description |
| :--- | :--- |
| `credit_card_transaction_all_amt_mean` | Average monthly credit card transaction amount. |
| `age` | Customer's age. |
| `credit_card_transaction_all_amt_trend` | Change in credit card transaction amount (Last 3M - First 3M). |
| `mobile_eft_all_amt_sum` | Total amount of mobile EFT transactions. |
| `active_months` | Number of unique active months.

The high importance of **trend features** confirms that monitoring changes in customer activity is crucial for accurate churn prediction.

## 7. Final Output

The model generated churn probabilities for the test data, which were saved to the file **'submission_optimized_xgb.csv'**. This file is the final deliverable.
