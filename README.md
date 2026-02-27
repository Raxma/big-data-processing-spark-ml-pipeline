This project applies PySpark to a warehouse-style sales dataset to build a modelling-ready analytics layer and predictive models. I segment transactions into High-Value vs Regular using a quantile threshold (rather than discarding valid high-value orders as “outliers”), then train a Spark ML Pipeline (encoding → feature assembly → Logistic Regression) to predict high-value customer behaviour. The notebook includes model confidence interpretation and additional regression work for car-ownership prediction to demonstrate end-to-end feature preparation, training, and evaluation.
# Big Data Processing (PySpark) — Customer Value Prediction & Segmentation

This project demonstrates an end-to-end big-data analytics workflow using **PySpark** to prepare warehouse-style datasets, apply **data quality controls**, and build **predictive models**.

The work focuses on two business questions:
1) **Segment transactions into High-Value vs Regular sales** (without deleting legitimate high-value orders as outliers).
2) **Predict High-Value customers** using a repeatable Spark ML pipeline.

A secondary modelling task is included:
- **Predict Number of Cars Owned** from customer demographics (using a scikit-learn regression workflow).

> **Portfolio note (KTP-ready evidence):** This repo shows practical skills in data integration, predictive analytics, and process discipline (cleaning, outlier policy decisions, feature engineering, evaluation).

---

## What I built

### 1) Data loading + integration (Spark)
- Created a Spark session and loaded multiple tables including `DimCustomer` and `FactInternetSales`.  
- Produced modelling-ready datasets by cleaning and joining tables into a final analytic DataFrame used for ML.  
(See notebook sections covering loading, cleaning, and joined dataset modelling.) 

### 2) Data quality and outlier handling (decision-driven)
- Audited missing values per table and removed irrelevant columns from customer data.   
- Detected outliers using IQR logic computed via Spark `approxQuantile`.   
- Applied **capping** for customer features like `YearlyIncome` rather than deleting records, preserving population realism.   
- For sales data, avoided removing extreme `SalesAmount` values because high-value purchases can be legitimate; instead used **segmentation**.   

### 3) High-value sales segmentation (Spark)
- Computed a **75th percentile threshold** of `SalesAmount` using `approxQuantile`.
- Created a `SaleType` label (`High-Value` vs `Regular`) using Spark `when/otherwise`.   

### 4) Predicting High-Value customers (Spark ML)
Built a repeatable Spark ML pipeline:
- `StringIndexer` + `OneHotEncoder` for categorical features
- `VectorAssembler` for feature vector
- `LogisticRegression` for classification
- Evaluated using:
  - **ROC AUC**
  - **Precision–Recall AUC**
  - Confusion matrix + Accuracy/Precision/Recall/F1 (via Pandas conversion)

This is implemented directly in the notebook pipeline section.   

### 5) Secondary task: Number of Cars Owned prediction (scikit-learn)
- Loaded customer dataset and engineered an ordinal encoding for `CommuteDistance`.
- Built a regression workflow with:
  - `ColumnTransformer` preprocessing
  - `LinearRegression` model
  - Evaluation using **RMSE** and **R²**
This is included as an additional predictive analytics example.   

---

## How to run (recommended)

### Option A — View results (no setup)
Open the notebook in GitHub and read the outputs directly:
- `notebooks/BigDataAssessment.ipynb`

### Option B — Run locally
#### 1) Create a virtual environment
```bash
python -m venv .venv
```
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
