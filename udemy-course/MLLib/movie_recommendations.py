from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType, IntegerType, LongType
from pyspark.ml.recommendation import ALS
import sys

def loadNames():
    movieNames = {}
    with open("datasets\\ml-100k\\u.item", "r") as f:
        for line in f:
            fields = line.split("|")
            movieNames[int(fields[0])] = fields[1]

    return movieNames

spark = SparkSession.builder.appName("ALSExample").getOrCreate()

# Create schema
ratingSchema = StructType([StructField("userID", IntegerType(), True),
                           StructField("movieID", IntegerType(), True),
                           StructField("rating", IntegerType(), True),
                           StructField("timestamp", LongType(), True)])

nameDict = loadNames()

ratings = spark.read.option("sep", "\t").schema(ratingSchema)\
            .csv("datasets\\ml-100k\\u.data")

print("Training ALS recommendation model...")

als = ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userID").setItemCol("movieID").setRatingCol("rating")

model = als.fit(ratings)

if (len(sys.argv) > 1):
    # Manually create a dataframe of the userID's we want to reccomentaion for
    userID = int(sys.argv[1])
    userSchema = StructType([StructField("userID", IntegerType(), True)])
    users = spark.createDataFrame([[userID,]], userSchema)

    recommendations = model.recommendForUserSubset(users, 10).collect()

    print("Top 10 recommendations for UserId: " + str(userID))

    for recs in recommendations:
        myRecs = recs[1]  #userRecs is (userID, [Row(movieId, rating), Row(movieID, rating)...])
        for rec in myRecs: #my Recs is just the column of recs for the user
            movie = rec[0] #For each rec in the list, extract the movie ID and rating
            rating = rec[1]
            movieName = nameDict[movie]
            print(movieName + str(rating))
