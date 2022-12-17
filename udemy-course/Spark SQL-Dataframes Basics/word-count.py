from pyspark.sql import SparkSession, functions as func

spark = SparkSession.builder.appName("WordCount").getOrCreate()

# Read each line of my book into dataframe
inputDF = spark.read.text("datasets\\Book.txt")

# Split using a reguler expression that extracts words
words = inputDF.select(func.explode(func.split(inputDF.value, "\\W+")).alias("word"))
words.filter(words.word != "")

# Normalize everything to lowercase
lowercaseWords = words.select(func.lower(words.word).alias("word"))

# Count up the occurences of each word
wordCounts = lowercaseWords.groupBy("word").count()
wordCounts.show()

# Soty by counts descending order
wordCountsSorted = wordCounts.sort(func.desc("count"))

# Show the result
wordCountsSorted.show()

spark.stop()