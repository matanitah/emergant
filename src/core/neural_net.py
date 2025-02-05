import numpy as np
class NeuralNetwork:
    def __init__(self):
        input_size = 8  # State observations
        hidden_size1 = 8  # Size of first hidden layer
        hidden_size2 = 8  # Size of second hidden layer
        output_size = 3  # [turn_left, turn_right, drop_pheromone]

        # Initialize a feedforward neural network with two hidden layers
        # The network architecture is: input_layer -> hidden_layer1 -> hidden_layer2 -> output_layer
        
        # weights1: Connects input layer to first hidden layer
        self.weights1 = np.random.randn(input_size, hidden_size1) * 0.1
        
        # weights2: Connects first hidden layer to second hidden layer
        self.weights2 = np.random.randn(hidden_size1, hidden_size2) * 0.1
        
        # weights3: Connects second hidden layer to output layer
        self.weights3 = np.random.randn(hidden_size2, output_size) * 0.1
        
    def forward(self, x):
        # Forward pass through the network
        # Uses tanh activation function for all layers
        
        # Store input for backpropagation
        self.input = x
        # Process through first hidden layer
        self.hidden1 = np.tanh(np.dot(x, self.weights1))
        # Process through second hidden layer
        self.hidden2 = np.tanh(np.dot(self.hidden1, self.weights2))
        # Process through output layer
        self.output = np.tanh(np.dot(self.hidden2, self.weights3))
        return self.output
    
    def update(self, learning_rate, gradient):
        # Full backpropagation through all layers
        # gradient is the initial error gradient at the output layer
        
        # Backpropagate through output layer to hidden2
        delta3 = gradient * (1 - np.square(self.output))  # derivative of tanh
        weights3_gradient = np.outer(self.hidden2, delta3)
        
        # Backpropagate through hidden2 to hidden1
        delta2 = np.dot(delta3, self.weights3.T) * (1 - np.square(self.hidden2))
        weights2_gradient = np.outer(self.hidden1, delta2)
        
        # Backpropagate through hidden1 to input
        delta1 = np.dot(delta2, self.weights2.T) * (1 - np.square(self.hidden1))
        weights1_gradient = np.outer(self.input, delta1)
        
        # Update all weights using their gradients
        self.weights3 += learning_rate * weights3_gradient
        self.weights2 += learning_rate * weights2_gradient
        self.weights1 += learning_rate * weights1_gradient