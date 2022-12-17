from pyspark.sql import SparkSession, functions as func
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

spark = SparkSession.builder.appName("MinTemperatures").getOrCreate()

schema = StructType([StructField("stationID", StringType(), True),
         StructField("date", IntegerType(), True),
         StructField("measure_type", StringType(), True),
         StructField("temperature", FloatType(), True)])

# Read the file as dataframe
df = spark.read.schema(schema).csv("datasets\\1800.csv")
df.printSchema()

# Filter with TMIN entries
minTemps = df.filter(df.measure_type == "TMIN")

# Select only stationID and temperature
stationTemps = minTemps.select("stationID", "temperature")

# Aggregate to find minimum temperature for every station
minTempsByStation = stationTemps.groupBy("stationID").min("temperature")
minTempsByStation.show()

# Convert temperature to fahrenheit and sort the dataset, add a new column named with temperature
minTempsByStationF = minTempsByStation.withColumn(
                    "temperature", func.round(func.col("min(temperature)") * 0.1 * (9.0 / 5.0) + 32.0 ,2))\
                    .select("stationID", "temperature").sort("temperature")

minTempsByStationF.show()

# Collect and print results
results = minTempsByStationF.collect()

for result in results:
    print(f"{result[0]}\t{result[1]:.2f}F")

pyspark.stop()