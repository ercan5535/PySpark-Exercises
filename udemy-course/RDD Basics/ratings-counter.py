from pyspark import SparkConf, SparkContext
import collections

conf = SparkConf().setMaster("local").setAppName("RatingsHistogram")
sc = SparkContext(conf = conf)

lines = sc.textFile("datasets\\ml-100k\\u.data")
ratings = lines.map(lambda x: x.split()[2])
result = ratings.countByValue()


sortedResults = sorted(result.items())
for key, value in sortedResults:
    print("%s %i" % (key, value))
