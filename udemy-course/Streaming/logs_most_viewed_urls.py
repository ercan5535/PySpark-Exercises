from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row, SparkSession, functions as func

# Create Spark Session
spark = SparkSession.builder.appName("StructuredStreaming").getOrCreate()

# Monitor the logs directory for new log data, in this case its logs_folder
accesLines = spark.readStream.text("Streaming\\logs_folder")

# Parse out the common log format to a DataFrame
contentSizeExp = r'\s(\d+)$'
statusExp = r'\s(\d{3})\s'
generalExp = r'\"(\S+)\s(\S+)\s*(\S*)\"'
timeExp = r'\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]'
hostExp = r'(^\S+\.[\S+\.]+\S+)\s'

logsDF = accesLines.select(func.regexp_extract('value', hostExp, 1).alias('host'),
                           func.regexp_extract('value', timeExp, 1).alias('timestamp'),
                           func.regexp_extract('value', generalExp, 1).alias('method'),
                           func.regexp_extract('value', generalExp, 2).alias('endpoint'),
                           func.regexp_extract('value', generalExp, 3).alias('protocol'),
                           func.regexp_extract('value', statusExp, 1).cast('integer').alias('status'),
                           func.regexp_extract('value', contentSizeExp, 1).cast('integer').alias('content_size'))

# Add current timestamp with eventTime
# it is a fake data to make window sliding on time series 
logsDF2 = logsDF.withColumn("eventTime", func.current_timestamp())

# Keep a running count of endpoints
endpointCounts = logsDF2.groupBy(func.window(func.col("eventTime"),
      windowDuration="30 seconds", slideDuration="10 seconds"), func.col("endpoint")).count()

sortedEndpointCounts = endpointCounts.orderBy(func.col("count").desc())

# Display the stream to the console
query = sortedEndpointCounts.writeStream.outputMode("complete").format("console") \
      .queryName("counts").start()

# Wait until we terminate the scripts
query.awaitTermination()