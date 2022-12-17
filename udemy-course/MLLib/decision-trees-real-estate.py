from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

# Create spark session
spark = SparkSession.builder.appName("DecisionTree").getOrCreate()

# Load data
data = spark.read.option("header", "true").option("inferSchema", "true")\
    .csv("datasets\\realestate.csv")

# A feature transformer that merges multiple columns into a vector column.
assembler = VectorAssembler().setInputCols(["HouseAge", "DistanceToMRT", \
                            "NumberConvenienceStores"]).setOutputCol("features")

# assembler.transform(data).show()
# +---+---------------+--------+-------------+-----------------------+--------+---------+---------------+--------------------+
# | No|TransactionDate|HouseAge|DistanceToMRT|NumberConvenienceStores|Latitude|Longitude|PriceOfUnitArea|            features|
# +---+---------------+--------+-------------+-----------------------+--------+---------+---------------+--------------------+
# |  1|       2012.917|    32.0|     84.87882|                     10|24.98298|121.54024|           37.9|[32.0,84.87882,10.0]|
# |  2|       2012.917|    19.5|     306.5947|                      9|24.98034|121.53951|           42.2| [19.5,306.5947,9.0]|
# |  3|       2013.583|    13.3|     561.9845|                      5|24.98746|121.54391|           47.3| [13.3,561.9845,5.0]|

# features column has HouseAge, DistanceToMRT, NumberConvenienceStores as list
# and our target column is PriceOfUnitArea
df = assembler.transform(data).select("PriceOfUnitArea", "features")

# Split train test data
trainTest = df.randomSplit([0.5, 0.5])
trainingDF = trainTest[0]
testDF = trainTest[1]

# Create model
dtr = DecisionTreeRegressor().setFeaturesCol("features").setLabelCol("PriceOfUnitArea")

# Train model
model = dtr.fit(trainingDF)

fullPredictions = model.transform(testDF).show()
# +---------------+-------------------+------------------+
# |PriceOfUnitArea|           features|        prediction|
# +---------------+-------------------+------------------+
# |           12.2|[30.9,6396.283,1.0]|              22.6|
# |           12.8|[16.5,4082.015,0.0]|              32.9|
# |           12.8|[32.0,1156.777,0.0]|14.533333333333333|