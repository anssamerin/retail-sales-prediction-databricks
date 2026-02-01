# Retail Sales Analysis & Prediction using PySpark and Databricks
Retail sales analysis and weekly sales prediction using PySpark and Databricks


## Project Overview
This project analyzes Walmart retail sales data and predicts weekly sales using PySpark ML on Databricks.

## Tools & Technologies
- Apache Spark (PySpark)
- Databricks Community Edition
- Python
- GitHub

## Key Features
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Weekly sales prediction using Linear Regression
- Model evaluation using RMSE
- Visualization using Databricks charts and exported results

## Dataset
Walmart historical sales dataset

## Output
- Sales trends
- Store-wise performance
- Actual vs predicted weekly sales

  ## Repository Structure

- `notebooks/` : Databricks PySpark implementation for data processing and model training
- `screenshots/` : Output visualizations used in the project report
- `README.md` : Project documentation

## Project Modules

- **01_data_loading_preprocessing.py**
  - Loads Walmart sales data into Spark DataFrame
  - Performs schema inference and basic validation

- **02_exploratory_data_analysis.py**
  - Store-wise sales analysis
  - Holiday vs non-holiday comparison
  - Monthly sales trend analysis

- **03_model_training.py**
  - Feature engineering using VectorAssembler
  - Train-test split
  - Linear Regression model training

- **04_model_evaluation_predictions.py**
  - Weekly sales prediction
  - Model evaluation using RMSE
  - Actual vs predicted sales comparison

    
## Execution

The project was developed and executed using Databricks Community Edition.
The workflow is organized into modular Python notebooks and should be executed
in the following order:

1. 01_data_loading_preprocessing.py  
2. 02_exploratory_data_analysis.py  
3. 03_model_training.py  
4. 04_model_evaluation_predictions.py

