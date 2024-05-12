import numpy as np 
import random 
import math

class NeuralNetwork:
    def __init__(self, size, hidden_layers=[]):
        """
        Initializing a neural network.
        The number of neurons will depend on the number of features in the dataset.
        For this first version of the lib, I am assuming that the dataset recieved has some x number of features.
        For the images the dataset will usually have each pixel as its own 'feature' (with the value being the brightness value) for each image.

        hidden_layers: defines the number of hidden layers. 
        it will have the format: [no_of_layers, no_of_neurons] where no_of_neurons applies to all the hidden layers created 
        """
        self.nn = []
        self.size = size

        if hidden_layers:
            for i in range(hidden_layers[0]):
                new_layer = []
                if i==0:
                    n_count = size
                else:
                    n_count = len(self.nn[i-1])
                # each hidden layer 
                for _ in range(hidden_layers[1]):
                    # initialize each neuron with random weights equal to number of neurons in the previous layer.
                    new_layer.append(
                        {
                            "weights": [random.random() for i in range(n_count)],
                            "bias": random.random(),
                            "act_val": None,
                        }
                    )
                self.nn.append(new_layer)
        
    
    def add_layer(self, no_of_neurons, activation_function='sigmoid' ,weight_initialization="default"):
        """
        
        """
        if len(self.nn)==0:
            n_count = self.size
        else:
            n_count = len(self.nn[-1])
        new_layer = []
        for _ in range(no_of_neurons):
            #weight initialization. NOTE: WORK IN PROGRESS

            weights = []
            if weight_initialization=="default":
                weights =[random.random() for i in range(n_count)]
                bias = random.random()
            elif weight_initialization =="xavier":
                #works best with sigmoid activation
                #calculated as a random number with a uniform probability distribution (U) between the range -(1/sqrt(n)) and 1/sqrt(n), where n is the number of inputs to the node source: https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
                factor = math.sqrt(1.0 / n_count)
                weights=[random.uniform(-factor, factor) for _ in range(n_count)]
                bias = random.uniform(-factor, factor)
            elif weight_initialization == 'He':
                #works best with relu activation
                factor = math.sqrt(2/n_count)
                #here i am using a uniform dist to draw the weights. NOTE: this code may change to draw weights directly from a Gaussian distribution as shown in the eq: calculated as a random number with a Gaussian probability distribution (G) with a mean of 0.0 and a standard deviation of sqrt(2/n), where n is the number of inputs to the node.
                weights=[random.uniform(-factor, factor) for _ in range(n_count)]
                bias = random.uniform(-factor, factor)

            
            new_layer.append(
                {
                    "weights": weights,
                    "bias": bias,
                    "act_val": None,
                }
            )
        
        self.nn.append(new_layer)

    def sigmoid(self, act_val, derivative=False):
        if derivative:
            return self.sigmoid_derivative(act_val)
        
        return 1 / (1 + np.exp(-1*act_val))

    def sigmoid_derivative(self, act_val):
        return (act_val * (1.0-act_val))
    
    def activate(self, x, neuron):
        
        act = 0
        for i in range(len(x)):
            act += (neuron['weights'][i] * x[i])
        
        act += neuron['bias']
        return self.sigmoid(act)
    
    def forward_pass(self, x):
        """
        Do calculation for the forward pass in order to get the proper activation values for the nodes in the hidden layers.
        """
        # first layer is input layer so it is ommitted from the loop
        # print(f"current network: {self.nn}\n")
        for layer in self.nn:
            
            neuron_act_vals = []
            for neuron in layer:
                # calculate the activation of each neuron in the current hidden layer using the prev layer's activation
                neuron["act_val"] = self.activate(x, neuron)
                neuron_act_vals.append(neuron["act_val"])
            
            x = neuron_act_vals
        
        return x
    
    def backward_prop(self, x, expected, lr):
        for i in reversed(range(len(self.nn))):
            layer = self.nn[i]
            layer_errors = list()
            #if its the last layer/output layer
            if i == len(self.nn)-1:
                for j in range(len(layer)):
                    neuron = layer[j]
                    #calculate error of output neuron by getting difference between true and predicted value
                    neuron_error = expected[j] - neuron['act_val']
                    layer_errors.append(neuron_error)
            #if its the hidden layer
            else:
                for j in range(len(layer)):
                    #backpropagating the errors from the output layer to the hidden layer
                    #neuron_error = sum(neuron["weights"]*neuron[error])
                    #since this is a reversed graph, we use the next neuron layer's weight list to get the weight connecting the current layer to the next
                    next_layer = self.nn[i+1]
                    neuron_error = 0
                    for neuron in next_layer:
                        neuron_error+=neuron['weights'][j]*neuron['error']
                    layer_errors.append(neuron_error)
            #update individual neuron errors
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['error'] = layer_errors[j]*self.sigmoid(neuron['act_val'], derivative=True)
        
        #update weights and biases
        #weight update = weight + (lr * inputs * neuron['error'])
        #bias update = bias + (lr*neuron['error'])
        for i in range(len(self.nn)):
            layer = self.nn[i]
            # print(f"layer currently at: {len(layer)}")
            for j in range(len(layer)):
                neuron = layer[j]
                # print(f"neuron currently at: {neuron}")
                for k in range(len(neuron['weights'])):
                    neuron['weights'][k] += (lr * x[k] * neuron['error'])
                neuron['bias']+=(lr*neuron['error'])
            x = [neuron['act_val'] for neuron in self.nn[i]]

    def train(self, X, Y, lr, epochs):
        """
        In this train function I am going to assign the first layer of neurons with the activation values from the corresponding features

        epochs: how many times you want to train the nn with the entire training set
        X: The feature dataset
        y: target column
        """

        # training on each record in the training set.
        # NOTE:for the time being we do not utilize the sgd mini batch training method
        print(f"neural network before training : {self.nn}")

        for k in range(1, epochs+1):
            total_loss = 0
            for i in range(len(X)):
                #assumption: the dataframe is converted to a list like format. 
                #NOTE: add iloc for dataframe support
                x = X[i]
                y = Y[i]
                #forward pass from the input layer to the output layer all done inside this function
                f_prop = self.forward_pass(x)
                print(f_prop)
                # print("-"*40)
                # print(f"f_prop: {f_prop}\n")
                #returning loss for each prediction
                # print(f"true values: {y}, f_prop values")
                total_loss += sum([(y[j] - f_prop[j]) ** 2 for j in range(len(f_prop))])
                # ---------------------------------- #
                self.backward_prop(x, y, lr)


            # print(self.nn[0])
            print(f"Epoch: {k}/{epochs} - Total Loss: {total_loss}")
        
    def desc(self):
        return self.nn
    
    def predict(self, inputs):
        output = self.forward_pass(inputs)
        return output
    


