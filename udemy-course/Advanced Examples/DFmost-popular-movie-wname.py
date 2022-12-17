from pyspark.sql import SparkSession, functions as func
from pyspark.sql.types import StructType, StructField, IntegerType, LongType

def loadMovieNames():
    movieNames = {}

    with open("datasets\\ml-100k\\u.item", "r") as f:
        for line in f:
            fields = line.split("|")
            movieNames[int(fields[0])] = fields[1]

    return movieNames

spark = SparkSession.builder.appName("MostPopularMovies").getOrCreate()

# Create broadcast object
nameDict = spark.sparkContext.broadcast(loadMovieNames())

# Create schema
schema = StructType([StructField("userID", IntegerType(), True),
                     StructField("movieID", IntegerType(), True),
                     StructField("rating", IntegerType(), True),
                     StructField("timestamp", LongType(), True)])

# Load up movie data as dataframe
moviesDF = spark.read.option("sep", "\t").schema(schema).csv("datasets\\ml-100k\\u.data")

# Group by movies with rating count  
TopMovies = moviesDF.groupBy("movieID").count()

# Create a user defined function
def lookupName(movieID):
    return nameDict.value[movieID]

lookupNameUDF = func.udf(lookupName)

# Add movieTitle column with movie names
moviesWithNames =  TopMovies.withColumn("movieTitle", lookupNameUDF(func.col("movieID")))

SortedMoviesWithNames = moviesWithNames.orderBy(func.desc("count"))

SortedMoviesWithNames.show(10, False)

spark.stop()
