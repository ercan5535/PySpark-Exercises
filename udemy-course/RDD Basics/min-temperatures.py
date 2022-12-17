from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("MinTemperatures")
sc = SparkContext(conf = conf)

def parseLine(line):
    fields = line.split(',')
    stationID = fields[0]
    entryType = fields[2]
    temperature = float(fields[3]) * 0.1 * (9.0 / 5.0) + 32.0
    return (stationID, entryType, temperature)

lines = sc.textFile("datasets\\1800.csv")
parsedLines = lines.map(parseLine) 
minTemps = parsedLines.filter(lambda x: "TMIN" in x[1])
# print(minTemps.take(3))
# [('ITE00100554', 'TMIN', 5.359999999999999), ('EZE00100082', 'TMIN', 7.699999999999999), ('ITE00100554', 'TMIN', 9.5)]
stationTemps = minTemps.map(lambda x: (x[0], x[2]))
# print(stationTemps.take(3))
# [('ITE00100554', 5.359999999999999), ('EZE00100082', 7.699999999999999), ('ITE00100554', 9.5)]
minTemps = stationTemps.reduceByKey(lambda x, y: min(x,y))
# print(minTemps.take(3))
# [('ITE00100554', 5.359999999999999), ('EZE00100082', 7.699999999999999)]
results = minTemps.collect()

for result in results:
    print(result[0] + "\t{:.2f}F".format(result[1]))
# ITE00100554     5.36F
# EZE00100082     7.70F