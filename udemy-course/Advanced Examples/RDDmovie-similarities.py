import sys
from pyspark import SparkConf, SparkContext
from math import sqrt

def filterDuplicates(line):
    (movie1, rating1) = line[1][0]
    (movie2, rating2) = line[1][1]
    return movie1 < movie2

def makePairs(line):
    (movie1, rating1) = line[1][0]
    (movie2, rating2) = line[1][1]
    return ((movie1, movie2), (rating1, rating2))

def computeCosineSimilarity(ratingPairs):
    numPairs = len(ratingPairs)
    sum_xx = sum_yy = sum_xy = 0
    score = 0

    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_xy += ratingX * ratingY
        sum_yy += ratingY * ratingY

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    if (denominator != 0):
        score = numerator / denominator

    return (score, numPairs)
    
def loadMovieNames():
    movieNames = {}
    with open("datasets\\ml-100k\\u.item", "r") as f:
        for line in f:
            fields = line.split("|")
            movieNames[int(fields[0])] = fields[1]

    return movieNames

conf = SparkConf().setMaster("local").setAppName("DegreesOfSeparation")
sc = SparkContext(conf=conf)

# Load movie names dict
nameDict = loadMovieNames()

# Load ratings data
data = sc.textFile("datasets\\ml-100k\\u.data")

# (userID, (movieID, rating))
ratings = data.map(lambda x: x.split("\t")).map(lambda x: (int(x[0]), (int(x[1]), int(x[2]))))

# join makes pair of every key values like (key, (value1, value2))
# to get rated movies by same person
joinedRatings = ratings.join(ratings)

# print(joinedRatings.filter(lambda x: x[0] == 196).take(3))
# [(196, ((242, 3), (242, 3))), (196, ((242, 3), (393, 4))), (196, ((242, 3), (381, 4)))]

# Drop duplicate pairs
filteredJoinedRatings = joinedRatings.filter(filterDuplicates)

# Get movie pairs as key and ratings as value
moviePairs = filteredJoinedRatings.map(makePairs)
# print(moviePairs.take(3))
# [((242, 393), (3, 4)), ((242, 381), (3, 4)), ((242, 251), (3, 3))]

# Get movie pairs and all the ratings between these pairs
# (movie1, movie2) => ((rating1, rating2), (rating1, rating2) ...)
moviePairRatings = moviePairs.groupByKey()

# Get movie similarity score and numPairs
# (movie1, movie2) => (score, numPairs)
moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity)
# print(moviePairSimilarities.take(2))
# [((242, 306), (0.9703755229417019, 34)), ((393, 663), (0.9346722344951355, 62))]


if (len(sys.argv) > 1):
    # Get user input as movieID
    movieID = int(sys.argv[1])

    # Set thresholds
    scoreThreshold = 0.90
    coOccurenceThreshold = 50

    # Filter for desired movieID and thresolds
    filteredResults = moviePairSimilarities.filter(lambda x: (x[0][0] == movieID or x[0][1] == movieID) and
                                    x[1][0] > scoreThreshold and x[1][1] > coOccurenceThreshold)

    # sort by score and take top 10
    results = filteredResults.sortBy(lambda x: x[1][0], ascending=False).take(10)
    # results = filteredResults.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending = False).take(10)

    print("top 10 similar movies for " + nameDict[movieID])
    for result in results:
        pair, sim = result
        # Display the similarity result that isn't the movie we're looking at
        # Choose the other movieID of pair1
        similarMovieID = pair[0]
        if (similarMovieID == movieID):
          similarMovieID = pair[1]

        print(f"{nameDict[similarMovieID]}\tscore: {sim[0]}\tstrength: {sim[1]}")

