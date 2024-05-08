import numpy as np
import random
import pandas as pd


class NeuralNetwork:
    def __init__(self, size, hidden_layers):
        """
        Initializing a neural network.
        The number of neurons will depend on the number of features in the dataset.
        For this first version of the lib, I am assuming that the dataset recieved has some x number of features.
        For the images the dataset will usually have each pixel as its own 'feature' (with the value being the brightness value) for each image.

        layers: defines the hidden layers and by default adds 16 neurons to each hidden layer
        """
        self.nn = []
        self.size = size
        # structure of the neurons : { weights=[], bias=0, act_val=0 }

        # initialize the first layer. this will have neurons equal to number of features in the dataset
        self.input_layer = [{"weights": [], "bias": 0, "act_val": 0}] * size
        self.nn.append(self.input_layer)
        # initialize the hidden layers
        for i in range(1, hidden_layers + 1):
            new_layer = []
            n_count = len(self.nn[i - 1])
            # each hidden layer has 16 neurons by default
            for _ in range(16):
                # initialize each neuron with random weights equal to number of neurons in the previous layer.
                new_layer.append(
                    {
                        "weights": [random.random()] * n_count,
                        "bias": random.random(),
                        "act_val": None,
                    }
                )
            self.nn.append(new_layer)

    def train(self, X, y=False):
        """
        In this train function I am going to assign the first layer of neurons with the activation values from the corresponding features

        X: The feature dataset
        y: target column
        """
        # training on each record in the training set.
        # NOTE:for the time being we do not utilize the sgd mini batch training method
        for i in range(len(X)):
            for j in range(len(self.nn[0])):
                self.nn[0][j]["act_val"] = df.iloc[i][j]

            self.forward_pass()
            print(f"Epoch: {i}/{len(X)}")

    def relu(self, act_val):
        return max(0, act_val)

    def activate(self, curr_neuron_index, neuron):
        return self.relu(
            np.dot(
                neuron["weights"],
                [i["act_val"] for i in self.nn[curr_neuron_index - 1]],
            )
        )
        # return neuron['act_val']

    def forward_pass(self):
        """
        Do calculation for the forward pass in order to get the proper activation values for the nodes in the hidden layers.
        """
        # first layer is input layer so it is ommitted from the loop
        for i in range(1, len(self.nn)):
            for neuron in self.nn[i]:
                # calculate the activation of each neuron in the current hidden layer using the prev layer's activation
                neuron["act_val"] = self.activate(i, neuron)

    def desc(self):
        return self.nn

    def __str__(self):
        pass


df = pd.read_csv("churn_data.csv")
# print(df)

# nn = NeuralNetwork(8, 2)
# nn.train()
