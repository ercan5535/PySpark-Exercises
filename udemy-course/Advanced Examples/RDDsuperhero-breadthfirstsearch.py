from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("DegreesOfSeparation")
sc = SparkContext(conf=conf)

# The characters we wish to find the degree of separation between:
startCharacterID = 5306 # Spiderman
targetCharacterID = 14 # ADAM 3,031

# Set the accumulator for 0, it is used as a counter
hitCounter = sc.accumulator(0)

def convertToBFS(line):
    fields = line.split()
    heroID = int(fields[0])
    connections = []
    for connection in fields[1:]:
        connections.append(int(connection))

    # Set default values for each node
    # WHITE means this node not touched before
    # Very high value for distance
    color = "WHITE"
    distance = 9999

    # We will start to check gray color as startCharacter
    if (heroID == startCharacterID):
        color = "GRAY"
        distance = 0

    return (heroID, (connections, distance, color))

def createStartingRDD():
    inputFile = sc.textFile("datasets\\Marvel+Graph.txt")
    return inputFile.map(convertToBFS)

def bfsMap(node):
    characterID = node[0]
    data = node[1]
    connections = data[0]
    distance = data[1]
    color = data[2]

    results = []

    # If this node needs to be expanded
    if (color == "GRAY"):
        for connection in connections:
            newCharacterID = connection
            newDistance = distance + 1
            # We touch this connection node, so color it gray
            newColor = "GRAY"
            if (targetCharacterID == connection):
                hitCounter.add(1)

            newEntry = (newCharacterID, ([], newDistance, newColor))
            results.append(newEntry)

        # We processed this node, so color it black
        color = "BLACK"

    # Emit the input node so we don't lose it
    results.append( (characterID, (connections, distance, color)) )
    return results

# Reduce flatten values by characterID
def bfsReduce(data1, data2):
    edges1 = data1[0]
    edges2 = data2[0]
    distance1 = data1[1]
    distance2 = data2[1]
    color1 = data1[2]
    color2 = data2[2]
    
    # Set initials
    distance = 9999
    color = color1
    edges = []

    # See if one is the original node with its connections.
    # If so preserve them.
    if (len(edges1) > 0):
        edges.extend(edges1)
    if (len(edges2) > 0):
        edges.extend(edges2)

    # Preserve minimum distance
    if (distance1 < distance):
        distance = distance1

    if (distance2 < distance):
        distance = distance2

    # Preserve darkest color
    if (color1 == 'WHITE' and (color2 == 'GRAY' or color2 == 'BLACK')):
        color = color2

    if (color1 == 'GRAY' and color2 == 'BLACK'):
        color = color2

    if (color2 == 'WHITE' and (color1 == 'GRAY' or color1 == 'BLACK')):
        color = color1

    if (color2 == 'GRAY' and color1 == 'BLACK'):
        color = color1

    return (edges, distance, color)


if __name__ == "__main__":
    iterationRdd = createStartingRDD()
    # print(iterationRdd.count())
    # print(iterationRdd.take(5))
    # .count() is necessary to make rdd is loaded
    # otherwise .take() will crash

    for iteration in range(0, 10):
        # Create new vertices as needed to darken or reduce distances in the
        # reduce stage. If we encounter the node we're looking for as a GRAY
        # node, increment our accumulator to signal that we're done.
        print("Runnigs BFS iteration" + str(iteration+1))
        mapped = iterationRdd.flatMap(bfsMap)

        # Note that mapped.count() action here forces the RDD to be evaluated, and
        # that's the only reason our accumulator is actually updated.
        print("Processing " + str(mapped.count()) + " values.")

        if (hitCounter.value > 0):
            print("Hit the target character! From " + str(hitCounter.value) \
                + " different direction(s).")
            break

        # Reducer combines flatten data for each character ID, preserving the darkest
        # color and shortest path.
        iterationRdd = mapped.reduceByKey(bfsReduce)
