import random
from math import sin

from matplotlib import pyplot as plt

from neuralNetwork import neuralNetwork

network = neuralNetwork(1, 20, 1, 0.4)

ACCEPTABLE_ERROR = 0.001
EVALUATE_ITERATIONS_COUNT = 1000

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
testTargets = targets[6000:]

plt.grid()
plt.title("Sinus")

plt.plot(values, targets, 'x', color='blue')
plt.draw()
plt.pause(5)

plt.plot(testData, testTargets, 'o', color='green')
plt.draw()
plt.pause(5)


error = 1
iteration = 0
networkLine = None
while error > ACCEPTABLE_ERROR:
    index = iteration % len(trainData)
    trainDataItem = trainData[index]
    trainTargetItem = trainTargets[index]
    network.train(trainDataItem, trainTargetItem)

    if iteration % EVALUATE_ITERATIONS_COUNT == 0:

        testDataHistory = []
        scorecad = []

        for testDataItem, testTargetItem in zip(testData, testTargets):
            networkOutput = network.query([testDataItem])[0]
            testDataHistory.append(networkOutput)
            diff = testTargetItem - networkOutput
            scorecad.append(diff)
        
        #compute error
        for errorVal in scorecad:
            error += abs(errorVal)
        error /= len(scorecad)

        print("iteration: ", iteration, "\terror: ", "%.4f" % error)
        if networkLine == None:
            line, = plt.plot(testData, testDataHistory, '*', color='purple')
            networkLine = line
        else:
            networkLine.set_xdata(testData)
            networkLine.set_ydata(testDataHistory)
        plt.pause(0.001)
    iteration += 1
    
plt.show()