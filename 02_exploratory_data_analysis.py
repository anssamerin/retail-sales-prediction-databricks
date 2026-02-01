# Databricks notebook source
# Databricks Notebook Source
# Module 2: Exploratory Data Analysis

from pyspark.sql.functions import *

# Store-wise sales analysis
df.groupBy("Store") \
  .agg(sum("Weekly_Sales").alias("Total_Sales")) \
  .orderBy("Total_Sales", ascending=False) \
  .display()

# Holiday vs Non-Holiday sales
df.groupBy("Holiday_Flag") \
  .agg(sum("Weekly_Sales").alias("Total_Sales")) \
  .display()

# Monthly sales trend
df.withColumn("Month", month("Date")) \
  .groupBy("Month") \
  .agg(sum("Weekly_Sales").alias("Sales")) \
  .orderBy("Month") \
  .display()
