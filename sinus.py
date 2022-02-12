import random
from math import sin

from matplotlib import pyplot as plt

from neuralNetwork import neuralNetwork

network = neuralNetwork(1, 10, 1, 0.4)

values = []
numValues = 7000

step = (2 * 3.1415) / numValues
x = 0
for i in range(numValues):
    values.append(x / (2 * 3.1415))
    x += step
random.shuffle(values)
print(values)


targets = []
for i in range(numValues):
    targets.append((0.5 * sin(values[i] * (2 * 3.1415))) + 0.5)

trainData = values[:6000]
trainTargets = targets[:6000]

testData = values[6000:]
testTargets = values[6000:]


for x in range(1000):
    for i in range(6000):
        network.train(trainData[i], trainTargets[i])
    print(str(x) + "%")

results = []
for i in range(1000):
    current = network.query(testData[i])
    results.append(current[0])

plt.plot(values, targets, 'x')
plt.plot(testData, results, 'o')
plt.show()
