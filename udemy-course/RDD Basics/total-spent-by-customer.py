from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("SpentByCustomer")
sc = SparkContext(conf = conf)

def ParseLine(line):
    fields = line.split(',')
    customerID = int(fields[0])
    amount = float(fields[2])
    return customerID, amount


input = sc.textFile("datasets\\customer-orders.csv")

parsedlines = input.map(ParseLine)
# print(parsedlines.take(3))
# [(44, 37.19), (35, 65.89), (2, 40.64)]

totalAmounts = parsedlines.reduceByKey(lambda x, y: x + y)
# print(totalAmounts.take(3))
# [(44, 4756.8899999999985), (35, 5155.419999999999), (2, 5994.59)]

totalAmountsSorted = totalAmounts.map(lambda x: (x[1], x[0])).sortByKey()

results = totalAmountsSorted.collect()
for result in results:
    print(f"Customer ID: {result[1]}, Total Amount: {result[0]}")