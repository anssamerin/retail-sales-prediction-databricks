# Databricks notebook source
# Databricks Notebook Source
# Module 3: Model Training

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

assembler = VectorAssembler(
    inputCols=["Temperature", "Fuel_Price", "CPI", "Unemployment", "Holiday_Flag"],
    outputCol="features"
)

df_features = assembler.transform(df)

train_df, test_df = df_features.randomSplit([0.8, 0.2], seed=42)

lr = LinearRegression(
    featuresCol="features",
    labelCol="Weekly_Sales"
)

model = lr.fit(train_df)
