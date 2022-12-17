from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

# Create spark session
spark = SparkSession.builder.appName("LinearRegression").getOrCreate()

# Load data
inputLines = spark.sparkContext.textFile("datasets\\regression.txt")
# Dense vectors are simply represented as NumPy array objects, 
# so there is no need to covert them for use in MLlib
data = inputLines.map(lambda x: x.split(",")).map(lambda x: (float(x[0]), Vectors.dense(float(x[1]))))

# Convert RDD to DataFrame
colNames = ["label", "features"]
df = data.toDF(colNames)

# Split main data to training and test data
trainTest = df.randomSplit([0.5, 0.5])
trainingDF = trainTest[0]
testDF = trainTest[1]

# Create linear regression model
lir = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Train model with training data
model = lir.fit(trainingDF)

# Generate predictions using our linear regression model for test dataframe
fullPredictions = model.transform(testDF).cache()
# fullPredictions = model.transform(testDF).show()
# +-----+--------+-------------------+
# |label|features|         prediction|
# +-----+--------+-------------------+
# |-3.74|  [3.75]|-2.6688580995870304|
# |-2.58|  [2.57]|-1.8258294474410537|
# |-2.54|  [2.39]|-1.6972318564357356|
# |-2.36|  [2.63]|-1.8686953111094933|
# |-2.17|  [2.19]|-1.5543456442076038|

results = fullPredictions.select("label", "prediction").show()

# Stop session
spark.stop()


