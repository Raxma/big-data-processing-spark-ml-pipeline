This project applies PySpark to a warehouse-style sales dataset to build a modelling-ready analytics layer and predictive models. I segment transactions into High-Value vs Regular using a quantile threshold (rather than discarding valid high-value orders as “outliers”), then train a Spark ML Pipeline (encoding → feature assembly → Logistic Regression) to predict high-value customer behaviour. The notebook includes model confidence interpretation and additional regression work for car-ownership prediction to demonstrate end-to-end feature preparation, training, and evaluation.

# Big Data Processing (PySpark) — Customer Value Segmentation & Prediction

This project uses **PySpark** on a warehouse-style sales dataset to build an analytics-ready modelling layer and train predictive models.

Core idea: instead of deleting legitimate high-value orders as “outliers”, I **segment transactions into High-Value vs Regular using a quantile threshold**, then train a repeatable **Spark ML Pipeline** to predict high-value customer behaviour.

## What’s inside
**Primary tasks**
1. **High-Value vs Regular segmentation (Spark)**
   - Computes a 75th percentile threshold of `SalesAmount` using `approxQuantile`
   - Creates a `SaleType` label via `when/otherwise`

2. **Predict High-Value customers (Spark ML)**
   - Pipeline: `StringIndexer` → `OneHotEncoder` → `VectorAssembler` → `LogisticRegression`
   - Evaluation: ROC AUC, PR AUC, confusion matrix, accuracy/precision/recall/F1 (via Pandas conversion)

**Secondary task**
3. **Predict Number of Cars Owned (scikit-learn regression)**
   - `ColumnTransformer` preprocessing + `LinearRegression`
   - Evaluation: RMSE and R²
   - Included to demonstrate end-to-end feature preparation + modelling beyond Spark

## Data quality & outlier policy (decision-driven)
- Audits missing values and removes irrelevant columns from customer data
- Detects outliers using IQR logic computed via Spark `approxQuantile`
- Caps features like `YearlyIncome` rather than deleting records (preserves population realism)
- For `SalesAmount`, avoids removal because extreme values can be legitimate → uses segmentation instead

## Project structure
notebooks/
BigDataAssessment.ipynb
data/
(place CSV files here - ignored by git)
requirements.txt
.gitignore
README.md


## Dataset setup
Place all source CSV files in:
- `data/` (recommended default)

If your dataset lives elsewhere, set an environment variable:

**PowerShell**
```powershell
$env:DATA_DIR="C:\path\to\your\data"
```
## CMD

set DATA_DIR=C:\path\to\your\data

## How to run
## Option A — View results (no setup)
Open the notebook in GitHub:

notebooks/BigDataAssessment.ipynb

## Option B — Run locally
Create + activate a virtual environment

python -m venv .venv
## Windows (PowerShell)

.\.venv\Scripts\Activate.ps1
## macOS/Linux

source .venv/bin/activate
## Install dependencies

pip install -r requirements.txt
## Launch the notebook

python -m notebook
Open: notebooks/BigDataAssessment.ipynb

## Portfolio note (KTP-ready evidence)
This repo demonstrates practical skills in:

data integration & modelling-ready dataset construction (joins, cleaning, quality checks)

predictive analytics (Spark ML + evaluation)

process discipline (explicit outlier policy, reproducible pipeline)


