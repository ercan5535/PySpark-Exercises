from pyspark.sql import SparkSession, functions as func
from pyspark.sql.types import StructType, StringType, StructField, IntegerType

spark = SparkSession.builder.appName("MostPopularHeroes").getOrCreate()

schema = StructType([StructField("id", IntegerType(), True),
                     StructField("name", StringType(), True)])

names = spark.read.schema(schema).option("sep", " ").csv("datasets\\Marvel+Names.txt")

lines = spark.read.text("datasets\\Marvel+Graph.txt")

connections = lines.withColumn("id", func.split(func.col("value"), " ")[0])\
                   .withColumn("numConnections", func.size(func.split(func.col("value"), " ")) - 1)\
                   .groupBy("id").agg(func.sum("numConnections").alias("numConnections"))


# Most Popular Hero:
mostPopular = connections.sort(func.col("numConnections").desc()).first()

# print(mostPopular)
# Row(id='859', numConnections=1937)

mostPopularName = names.filter(func.col("id") == mostPopular[0]).select("name").first()
print(f"{mostPopularName[0]} is the most popular superhero with {mostPopular[1]} connections")

# Least Popular Heroes:
minConnection = connections.select(func.min(func.col("numConnections"))).collect()[0][0]
LeastPopularHeroesIDs = connections.filter(func.col("numConnections") == minConnection)

LeastPopularHeroesIDsWithNames = LeastPopularHeroesIDs.join(names, "id")
print("Least Popular Heroes:")
LeastPopularHeroesIDsWithNames.select("name").show()