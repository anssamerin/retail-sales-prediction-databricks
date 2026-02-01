# Databricks notebook source
# Databricks Notebook Source
# Module 1: Data Loading and Preprocessing

from pyspark.sql.functions import *

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv('/Workspace/DATA/Walmart.csv')

df.printSchema()
df.show()
