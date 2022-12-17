import re
from pyspark import SparkConf, SparkContext

def normalizeWords(text):
    return re.compile(r"\W+", re.UNICODE).split(text.lower())

conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf = conf)

input = sc.textFile("datasets\\Book.txt")
words = input.flatMap(normalizeWords)
# print(words.take(3))
# ['self', 'employment', 'building']

wordCounts = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
# print(wordCounts.take(3))
# [('self', 111), ('employment', 75), ('building', 33)]

wordCountsSorted = wordCounts.map(lambda x: (x[1], x[0])).sortByKey()
# print(wordCountsSorted.take(3))
# [(1, 'achieving'), (1, 'contents'), (1, 'preparation')]

results = wordCountsSorted.collect()

for result in results:
    count = str(result[0])
    word = result[1]
    if word:
        print(word + ":\t\t" + count)