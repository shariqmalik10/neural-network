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
        #current_idx refers to the latest layer in the network
        self.current_idx = 0
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

        self.current_idx = hidden_layers

    def add_layer(self, no_of_neurons):
        # initializing the output layer - for now we keep it to multi class classification. for that we will calculate the number of unique values in the target column and create that many neurons in the output layer

        n_count = len(self.nn[self.current_idx])
        new_layer = []
        for _ in range(no_of_neurons):
            new_layer.append(
                {
                    "weights": [random.random()] * n_count,
                    "bias": random.random(),
                    "act_val": None,
                }
            )
        
        self.nn.append(new_layer)
        self.current_idx = len(self.nn)-1

    def train(self, X, y, epochs=5):
        """
        In this train function I am going to assign the first layer of neurons with the activation values from the corresponding features

        epochs: how many times you want to train the nn with the entire training set
        X: The feature dataset
        y: target column
        """

        # training on each record in the training set.
        # NOTE:for the time being we do not utilize the sgd mini batch training method

        for k in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                # x = X[i]
                for j in range(len(self.nn[0])):
                    # print(f"update of input layer triggered ! {j}")
                    # print(f"df vals: {df.iloc[i][j]}")
                    self.nn[0][j]["act_val"] = X.iloc[i][j]
                
                # print(X[i])
                #forward pass from the input layer to the output layer all done inside this function
                # f_prop = self.forward_pass(x)
                f_prop = self.forward_pass()
                # print(f"f_prop: {f_prop}\n")
                soft = self.softmax(f_prop)
                # print(f"softmax val: {soft}")
                #loss compute. since we have a classification problem we will use ce-loss
                # print(f"y val: {y[i]}")
                loss = self.ce_loss(soft, y[i])
                # print(f"ce-loss: {loss}")
                total_loss += loss
                # print("-"*20)

            # print(self.nn[0])
            print(f"Epoch: {k+1}/{epochs} - loss: {total_loss}")
            # print(f"State of nn: {self.nn}")

    def relu(self, act_val):
        return max(0, act_val)
        
    def softmax(self, x):
        """
        
        """
        probs = np.exp(x -np.max(x))
        pred = probs/np.sum(probs)
        return pred
    
    def ce_loss(self, y_hat, y_pred):
        """
        returns the cross-entropy loss
        y_hat: entries in the ground truth label 
        y_pred: entries in pred vector
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(np.log(y_pred) * y_hat)

        #epsilon is added to avoid the possibility of getting a inf result in the 

    
    def decide(self, arr):
        return arr.index(max(arr))

    def activate(self, curr_neuron_index, neuron):
        # return self.relu(
        #     np.dot(
        #         neuron["weights"],
        #         x,
        #     )
        # )
        return self.relu(
            np.dot(
                neuron["weights"],
                [i["act_val"] for i in self.nn[curr_neuron_index - 1]],
            )
        )
    

    def forward_pass(self, x=False):
        """
        Do calculation for the forward pass in order to get the proper activation values for the nodes in the hidden layers.
        """
        # first layer is input layer so it is ommitted from the loop
        for i in range(1, len(self.nn)):
            neuron_act_vals = []
            for neuron in self.nn[i]:
                # calculate the activation of each neuron in the current hidden layer using the prev layer's activation
                neuron["act_val"] = self.activate(i, neuron)
                # neuron_act_vals.append(neuron["act_val"])
                # act = neuron["act_val"]
                # print(f"neuron activation value calculated: {act}")
            # x = neuron_act_vals
        # print([i['act_val'] for i in self.nn[len(self.nn)-1]])
        return [i['act_val'] for i in self.nn[len(self.nn)-1]]

    def desc(self):
        return self.nn

    def __str__(self):
        pass


df = pd.read_csv("churn_data.csv")
# print(df)

# nn = NeuralNetwork(8, 2)
# nn.train()
