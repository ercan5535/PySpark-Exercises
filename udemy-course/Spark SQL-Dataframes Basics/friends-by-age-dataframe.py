from pyspark.sql import SparkSession, functions as func

spark = SparkSession.builder.appName("FriendsByAge").getOrCreate()

lines = spark.read.option("header", "true").option("inferSchema", "true")\
    .csv("datasets\\fakefriends-header.csv")

# Select only age and numFriends column
friendsByAge = lines.select("age", "friends")

# Group by age and then compute average
friendsByAge.groupBy("age").avg("friends").show()

# Sorted
friendsByAge.groupBy("age").avg("friends").sort("age").show()

# Formatted more nicely
friendsByAge.groupBy("age").agg(func.round(func.avg("friends"), 2)).sort("age").show()

# With a custom column name
friendsByAge.groupBy("age").agg(func.round(func.avg("friends"), 2).alias("friends_avg")).sort("age").show()

spark.stop()