import numpy as np
import random

class NeuralNetwork:
    def __init__(self, size, layers):
        """
        Initializing a neural network. 
        The number of neurons will depend on the number of features in the dataset. 
        For this first version of the lib, I am assuming that the dataset recieved has some x number of features (not containing any images or videos since there is additional preprocessing to do for those data types)

        layers: defines the hidden layers and by default adds 16 neurons to each hidden layer
        """
        self.initial_layer = []
        self.layers = []
        

        #populate the initial layer with all the neurons and the bias(which will be random value initially)
        self.initial_layer = [Neuron()]*size
        self.layers.append(self.initial_layer)

        #create the hidden layers
        for i in range(layers):
            new_layer = [Neuron()]*16
            self.layers.append(new_layer)
        
    def forward_pass(self, features):
        """
        Do calculation for the forward pass in order to get the proper activation values for the nodes in the hidden layers
        """
        for i in range(1,self.layers):
            for j in self.layers[i]:
                #calculate the activation of each neuron in the current hidden layer using the prev layer's activation
                pass
                    





    def desc(self):
        return len(self.initial_layer)
    
    def __str__(self):
        pass


class Neuron:
    def __init__(self, act=None, new=False):
        """
        initializing neuron with activation value. (also include weight and bias) \n
        the neuron is initialized with random weight value

        new: just a boolean value to keep track of when the neuron is for initial layer
        """
        self.weight = random.random()
        self.bias = random.random()

        if act:
            #initialize the activation vals for them
            self.act = act
        


nn = NeuralNetwork(784)