import numpy as np
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot


# ensure the plots are inside this notebook, not an external window

# neural network class definition


class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes

        # learning rate
        self.lr = learningRate

        # link weight matrices, wih and who
        # wieghts inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        # self.wih = (np.random.rand(self.hnodes, self.inodes) -0.5 )
        # self.who = (np.random.rand(self.onodes, self.hnodes) -0.5 )

        # optionale Änderung
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.bias_hidden = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, 1))
        self.bias_output = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, 1))

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        # self.activation_function = lambda x: activationFunction.sigmoid(x)

        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs + self.bias_hidden)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs + self.bias_output)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers

        gradient_output = output_errors * final_outputs * (- final_outputs + 1.0)
        self.who += self.lr * np.dot(gradient_output, np.transpose(hidden_outputs))
        self.bias_output += self.lr * gradient_output


        # update the weights for the links between the input and hidden layers

        gradient_hidden = hidden_errors * hidden_outputs * (- hidden_outputs + 1.0)
        self.wih += self.lr * np.dot(gradient_hidden, np.transpose(inputs))
        self.bias_hidden += self.lr * gradient_hidden




        pass

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculatesignals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs + self.bias_hidden )

        # calculate signals into final ouput layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate zje signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs + self.bias_output )

        return final_outputs
