from pyspark.sql import SparkSession, functions as func
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

spark = SparkSession.builder.appName("TotalSpentByCustomer").getOrCreate()

schema = StructType([StructField("customerID", IntegerType(), True),
         StructField("transactionID", IntegerType(), True),
         StructField("amount", FloatType(), True)])

# Read file as dataframe
df = spark.read.schema(schema).csv("datasets\\customer-orders.csv")
df.printSchema()

customerGrouped = df.groupBy("customerID").agg(func.round(func.sum("amount"), 2).alias("total_amount")).sort("total_amount")
customerGrouped.show(customerGrouped.count())

spark.stop()

