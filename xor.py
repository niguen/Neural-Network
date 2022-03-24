from neuralNetwork import neuralNetwork

network = neuralNetwork(2, 10, 1, 0.2)

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]


print("before Training:")

print("Expected result:" + str(outputs[0]))
print(network.query(inputs[0]))

print("Expected result:" + str(outputs[1]))
print(network.query(inputs[1]))

print("Expected result:" + str(outputs[2]))
print(network.query(inputs[2]))

print("Expected result:" + str(outputs[3]))
print(network.query(inputs[3]))

network.train(inputs[0], outputs[0])

for i in range(20000):
    for record in range(len(inputs)):
        network.train(inputs[record], outputs[record])


print("After Training:")

print("Expected result:" + str(outputs[0]))
print(network.query(inputs[0]))

print("Expected result:" + str(outputs[1]))
print(network.query(inputs[1]))

print("Expected result:" + str(outputs[2]))
print(network.query(inputs[2]))

print("Expected result:" + str(outputs[3]))
print(network.query(inputs[3]))


