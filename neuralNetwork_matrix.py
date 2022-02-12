from matrix import Matrix


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

        self.wih = Matrix(self.inodes, self.hnodes).random(self.hnodes).T()
        self.who = Matrix(self.hnodes, self.onodes).random(self.hnodes).T()

        # activation function is the sigmoid function
        # self.activation_function = lambda x: scipy.special.expit(x)

        self.activation_function = lambda x: Matrix.sigmoid(x)

        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array

        if type(inputs_list) is float or type(inputs_list) is int:
            values = [inputs_list]
            inputs = Matrix(1, 1, values)
        else:
            inputs = Matrix(len(inputs_list), 1, inputs_list)

        if type(targets_list) is float or type(targets_list) is int:
            values = [targets_list]
            targets = Matrix(1, 1, values)
        else:
            targets = Matrix(len(targets_list), 1, targets_list)


        # calculate signals into hidden layer
        hidden_inputs = Matrix.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = Matrix.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output layer
        final_outputs = final_inputs

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = Matrix.dot(Matrix.transpose(self.who), output_errors)

        # update the weights for the links between the hidden and output layers
        who1 = (output_errors * final_outputs * (- final_outputs + 1.0))

        self.who += Matrix.dot((output_errors * final_outputs * (- final_outputs + 1.0)), Matrix.transpose(hidden_outputs)) * self.lr

        # update the weights for the links between the input and hidden layers
        value2 = (hidden_errors * hidden_outputs * (- hidden_outputs + 1.0))
        self.wih += Matrix.dot((hidden_errors * hidden_outputs * (- hidden_outputs + 1.0)), Matrix.transpose(inputs)) * self.lr


        pass

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        if type(inputs_list) is float or type(inputs_list) is int:
            values = [inputs_list]
            inputs = Matrix(1, 1, values)
        else:
            inputs = Matrix(len(inputs_list), 1, inputs_list)

        # calculatesignals into hidden layer
        hidden_inputs = Matrix.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final ouput layer
        final_inputs = Matrix.dot(self.who, hidden_outputs)

        # calculate zje signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
