# Databricks notebook source
# Databricks Notebook Source
# Module 4: Model Evaluation and Predictions

from pyspark.ml.evaluation import RegressionEvaluator

# -------------------------------------------------------
# Generate Predictions on Test Dataset
# -------------------------------------------------------
predictions = model.transform(test_df)

# Rename prediction column for clarity
predictions = predictions.withColumnRenamed(
    "prediction",
    "next_week_prediction"
)

# Display Actual vs Predicted Weekly Sales
predictions.select(
    "Weekly_Sales",
    "next_week_prediction"
).display()

# -------------------------------------------------------
# Model Evaluation using RMSE
# -------------------------------------------------------
evaluator = RegressionEvaluator(
    labelCol="Weekly_Sales",
    predictionCol="next_week_prediction",
    metricName="rmse"
)

rmse = evaluator.evaluate(predictions)
print("Root Mean Square Error (RMSE):", rmse)

# -------------------------------------------------------
# Optional: Additional Metrics (R2)
# -------------------------------------------------------
evaluator_r2 = RegressionEvaluator(
    labelCol="Weekly_Sales",
    predictionCol="next_week_prediction",
    metricName="r2"
)

r2 = evaluator_r2.evaluate(predictions)
print("R2 Score:", r2)
