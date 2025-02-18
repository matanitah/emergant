import numpy as np
class NeuralNetwork:
    def __init__(self, hidden_sizes):
        """
        Initialize a feedforward neural network with variable hidden layers.
        
        Args:
            layer_sizes (list): List of integers representing the size of each layer.
                              First element is input size, last element is output size,
                              and elements in between are hidden layer sizes.
        """
        self.hidden_sizes = hidden_sizes
        self.layer_sizes = [8] + hidden_sizes + [2]  # 8 inputs, 2 outputs
        self.num_layers = len(self.layer_sizes)
        
        # Initialize weights between all layers
        self.weights = []
        for i in range(self.num_layers - 1):
            # Initialize weights with small random values
            self.weights.append(
                np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.1
            )
        
        # Storage for layer activations during forward pass
        self.activations = []
    
    def forward(self, x):
        # Store input
        self.activations = [x]
        
        # Forward propagate through each layer
        current_activation = x
        for weights in self.weights:
            current_activation = np.tanh(np.dot(current_activation, weights))
            self.activations.append(current_activation)
            
        return self.activations[-1]
    
    def update(self, learning_rate, gradient):
        # Start with output gradient
        delta = gradient * (1 - np.square(self.activations[-1]))
        
        # Backpropagate through each layer
        weight_gradients = []
        for layer_idx in reversed(range(len(self.weights))):
            # Calculate weight gradients
            weight_gradient = np.outer(self.activations[layer_idx], delta)
            weight_gradients.insert(0, weight_gradient)
            
            # Calculate delta for next layer
            if layer_idx > 0:  # No need to calculate delta for input layer
                delta = np.dot(delta, self.weights[layer_idx].T) * (1 - np.square(self.activations[layer_idx]))
        
        # Update all weights using their gradients
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * weight_gradients[i]