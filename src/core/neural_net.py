import numpy as np
class NeuralNetwork:
    def __init__(self):
        input_size = 7  # State observations
        hidden_size = 8 # Size of hidden layer
        output_size = 3  # [turn_left, turn_right, drop_pheromone]

        # Initialize a simple feedforward neural network with one hidden layer
        # The network architecture is: input_layer -> hidden_layer -> output_layer
        
        # weights1: Connects input layer to hidden layer
        # Initialized with small random values (scaled by 0.1) to prevent extreme initial behaviors
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        
        # weights2: Connects hidden layer to output layer
        # Initialized with small random values (scaled by 0.1) to prevent extreme initial behaviors
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.1
        
    def forward(self, x):
        # Forward pass through the network
        # Uses tanh activation function for both layers to output values between -1 and 1
        
        # Process through hidden layer
        self.hidden = np.tanh(np.dot(x, self.weights1))
        # Process through output layer
        self.output = np.tanh(np.dot(self.hidden, self.weights2))
        return self.output
    
    def update(self, learning_rate, gradient):
        # Simple gradient update for reinforcement learning
        # Only updates the output layer weights (weights2) based on the reward signal
        # learning_rate: Controls how much the weights change in response to errors
        # gradient: Computed based on the reward history and action history
        self.weights2 += learning_rate * gradient