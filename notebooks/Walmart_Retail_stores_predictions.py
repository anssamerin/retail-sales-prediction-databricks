# Databricks notebook source
# DBTITLE 1,Cell 2
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv('/Workspace/DATA/Walmart.csv')


# COMMAND ----------

# DBTITLE 1,displaying data
df.show()

# COMMAND ----------

# DBTITLE 1,Basic info
df.printSchema()
df.count()

# COMMAND ----------

# DBTITLE 1,3. Check Missing Values
from pyspark.sql.functions import col, sum

df.select([
    sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]).show()


# COMMAND ----------

# DBTITLE 1,Handle Nulls
df_clean = df.dropna()


# COMMAND ----------

from pyspark.sql.functions import when, col

df_clean = df_clean.withColumn(
    "Store_Name",
    when(col("Store") == 1, "Walmart – Chennai Central")
    .when(col("Store") == 2, "Walmart – Bangalore Whitefield")
    .when(col("Store") == 3, "Walmart – Hyderabad Gachibowli")
    .when(col("Store") == 4, "Walmart – Mumbai Andheri")
    .when(col("Store") == 5, "Walmart – Delhi Connaught Place")
    .when(col("Store") == 6, "Walmart – Pune Hinjewadi")
    .when(col("Store") == 7, "Walmart – Kochi Edappally")
    .when(col("Store") == 8, "Walmart – Trivandrum Kazhakkoottam")
    .when(col("Store") == 9, "Walmart – Coimbatore RS Puram")
    .when(col("Store") == 10, "Walmart – Madurai Anna Nagar")
    .when(col("Store") == 11, "Walmart – Trichy Cantonment")
    .when(col("Store") == 12, "Walmart – Salem Five Roads")
    .when(col("Store") == 13, "Walmart – Erode Perundurai")
    .when(col("Store") == 14, "Walmart – Tiruppur Avinashi Road")
    .when(col("Store") == 15, "Walmart – Vellore Katpadi")
    .when(col("Store") == 16, "Walmart – Hosur SIPCOT")
    .when(col("Store") == 17, "Walmart – Noida Sector 18")
    .when(col("Store") == 18, "Walmart – Gurgaon Cyber Hub")
    .when(col("Store") == 19, "Walmart – Faridabad Sector 15")
    .when(col("Store") == 20, "Walmart – Jaipur MI Road")
    .when(col("Store") == 21, "Walmart – Ahmedabad SG Highway")
    .when(col("Store") == 22, "Walmart – Surat Adajan")
    .when(col("Store") == 23, "Walmart – Vadodara Alkapuri")
    .when(col("Store") == 24, "Walmart – Indore Vijay Nagar")
    .when(col("Store") == 25, "Walmart – Bhopal MP Nagar")
    .otherwise("Walmart – Rest of india ")
)


# COMMAND ----------

dashboard_df = df_clean.groupBy("Store_Name") \
    .sum("Weekly_Sales") \
    .orderBy("sum(Weekly_Sales)", ascending=False)

dashboard_df.display()


# COMMAND ----------

# DBTITLE 1,Convert Date Column
from pyspark.sql.functions import to_date

df_clean = df_clean.withColumn(
    "Date",
    to_date("Date", "dd-MM-yyyy")
)


# COMMAND ----------

# DBTITLE 1,Create Time Features
from pyspark.sql.functions import year, month, weekofyear

df_clean = df_clean \
    .withColumn("Year", year("Date")) \
    .withColumn("Month", month("Date")) \
    .withColumn("Week", weekofyear("Date"))


# COMMAND ----------

# DBTITLE 1,Total Sales
from pyspark.sql.functions import sum

df_clean.agg(sum("Weekly_Sales").alias("Total_Sales")).show()


# COMMAND ----------

# DBTITLE 1,Sales by Store
df_clean.groupBy("Store_Name") \
    .agg(sum("Weekly_Sales").alias("Store_Sales")) \
    .orderBy("Store_Sales", ascending=False) \
    .display()


# COMMAND ----------

# DBTITLE 1,Holiday vs Non-Holiday
df_clean.groupBy("Holiday_Flag") \
    .agg(sum("Weekly_Sales").alias("Sales")) \
    .display()


# COMMAND ----------

# DBTITLE 1,Best Performing Month
df_clean.groupBy("Month") \
    .agg(sum("Weekly_Sales").alias("Sales")) \
    .orderBy("Sales", ascending=False) \
    .display()


# COMMAND ----------

# DBTITLE 1,Temperature Impact
df_clean.select("Temperature", "Weekly_Sales").display()


# COMMAND ----------

# DBTITLE 1,Feature Preparation for ML
from pyspark.ml.feature import VectorAssembler

feature_cols = [
    "Temperature", "Fuel_Price", "CPI",
    "Unemployment", "Holiday_Flag", "Month"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

data_ml = assembler.transform(df_clean) \
    .select("features", "Weekly_Sales")


# COMMAND ----------

# DBTITLE 1,Train-Test Split
train_df, test_df = data_ml.randomSplit([0.8, 0.2], seed=42)


# COMMAND ----------

# DBTITLE 1,train model
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(labelCol="Weekly_Sales")
model = lr.fit(train_df)



# COMMAND ----------

# DBTITLE 1,prediction
predictions = model.transform(test_df)
predictions = predictions.withColumnRenamed(
    "prediction",
    "next_week_prediction"
)
predictions.select("Weekly_Sales", "next_week_prediction").display()


# COMMAND ----------

# DBTITLE 1,Model Evaluation
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(
    labelCol="Weekly_Sales",
    predictionCol="next_week_prediction",
    metricName="rmse"
)

rmse = evaluator.evaluate(predictions)
rmse


# COMMAND ----------

spark.createDataFrame(
    [("RMSE", rmse)],
    ["Metric", "Value"]
).display()


# COMMAND ----------

from pyspark.sql.functions import sum

dashboard_df = df_clean.groupBy("Date", "Store_Name", "Holiday_Flag") \
    .agg(sum("Weekly_Sales").alias("Total_Sales"))

dashboard_df.createOrReplaceTempView("retail_dashboard")


# COMMAND ----------

dashboard_df.display()


# COMMAND ----------

# DBTITLE 1,Cell 24
final_df = test_df.join(df_clean.select("Date", "Store_Name"), on=["Date"], how="left")
final_df = final_df.select(
    "Date",
    "Store_Name",
    "Weekly_Sales",
    "next_week_prediction"
)


# COMMAND ----------

final_df.display()

# COMMAND ----------

final_df.coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv('/Workspace/OUTPUT/test.csv')
