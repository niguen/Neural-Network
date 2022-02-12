from math import sin

import matplotlib.pyplot as plt

from neuralNetwork_matrix import neuralNetwork
from matplotlib import pyplot as plt

network = neuralNetwork(1, 20, 1, 0.5)

values = []
numValues = 10

step = (2 * 3.1415) / numValues
x = 0
for i in range(numValues):
    values.append(x)
    x += step

for x in range(10000):
    for i in range(numValues):
        current = values[i]
        network.train(current, sin(current))
    print(str(x) + "%")

results = []
expected = []
for i in range(numValues):
    current = network.query(values[i])
    results.append(current.values[0])
    expected.append(sin(values[i]))

plt.plot(values, expected)
plt.plot(values, results)
plt.show()
