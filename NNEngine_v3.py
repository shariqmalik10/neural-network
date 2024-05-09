import numpy as np
import random
import pandas as pd


class NeuralNetwork:
    def __init__(self, size, hidden_layers, output):
        """
        Initializing a neural network.
        The number of neurons will depend on the number of features in the dataset.
        For this first version of the lib, I am assuming that the dataset recieved has some x number of features.
        For the images the dataset will usually have each pixel as its own 'feature' (with the value being the brightness value) for each image.

        layers: defines the hidden layers and by default adds 16 neurons to each hidden layer
        output: number of outputs
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
                        "weights": [random.random() for i in range(n_count)],
                        "bias": random.random(),
                        "act_val": None,
                    }
                )
            self.nn.append(new_layer)
        
        #add the output layer
        n_count = len(self.nn[len(self.nn)-1])
        new_layer = []
        for _ in range(output):
            new_layer.append(
                {
                    "weights": [random.random() for i in range(n_count)],
                    "bias": random.random(),
                    "act_val": None,
                }
            )
        
        self.nn.append(new_layer)

        self.current_idx = len(self.nn)-1

    def add_layer(self, no_of_neurons):
        # initializing the output layer - for now we keep it to multi class classification. for that we will calculate the number of unique values in the target column and create that many neurons in the output layer

        n_count = len(self.nn[self.current_idx])
        new_layer = []
        for _ in range(no_of_neurons):
            new_layer.append(
                {
                    "weights": [random.random() for i in range(n_count)],
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
            for i in range(5):
                x = X.iloc[i]
                #forward pass from the input layer to the output layer all done inside this function
                f_prop = self.forward_pass(x)
                # print(f"f_prop: {f_prop}\n")

                # ---------------------------------- #
                self.backward_propagation(x, y, lr)


                

            # print(self.nn[0])
            print(f"Epoch: {k+1}/{epochs}")
            

    def relu(self, act_val):
        return max(0, act_val)

    def sigmoid(self, act_val, derivative=False):
        if derivative:
            self.sigmoid_derivative(act_val)
        
        return 1.0 / (1.0 + np.exp(-act_val))

    def sigmoid_derivative(self, act_val):
        return act_val * (1.0-act_val)
        
    def softmax(self, x):
        """
        
        """
        probs = np.exp(x -np.max(x))
        pred = probs/np.sum(probs)
        return pred

    def activate(self, x, neuron):
        
        act = neuron['bias']
        for i in range(len(neuron['weights'])):
            act+= (neuron['weights'][i] * x[i])

        return self.sigmoid(act)

    def forward_pass(self, x):
        """
        Do calculation for the forward pass in order to get the proper activation values for the nodes in the hidden layers.
        """
        # first layer is input layer so it is ommitted from the loop
        print(f"starting act_vals: {x}")
        for i in range(1, len(self.nn)):
            neuron_act_vals = []
            for neuron in self.nn[i]:
                # calculate the activation of each neuron in the current hidden layer using the prev layer's activation
                neuron["act_val"] = self.activate(x, neuron)
                neuron_act_vals.append(neuron["act_val"])

            print(f"Neuron value: {neuron_act_vals}")
            print("-"*30)
                # act = neuron["act_val"]
                # print(f"neuron activation value calculated: {act}")
            x = neuron_act_vals
        # print([i['act_val'] for i in self.nn[len(self.nn)-1]])

        print(f"final act_vals: {x} \n" )
        return x
    
    def backward_propagation(self, X, expected, lr):
        #backprop starting from the final layer 
        for i in reversed(range(1, len(self.nn))):
            layer = self.nn[i]
            layer_errors = list()
            if i == len(self.nn) - 1:
                for j in range(len(layer)):
                    neuron = layer[j]
                    #calculate error of output neuron by getting difference between true and predicted value
                    error = expected[j] - neuron['act_val']
                    layer_errors.append(error)
            
            else:
                #for all the hidden layer neurons get the error by using the eq: neuron_weights * neuron['error']
                for j in range(len(layer)):
                    neuron_error = sum([neuron['weights'][j] *neuron['error'] for neuron in self.nn[i+1]])
                    layer_errors.append(neuron_error)
            
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['error'] = layer_errors[j] * self.sigmoid(neuron['act_val'], derivative=True)
                
                



                


    def desc(self):
        return self.nn