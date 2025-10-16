import numpy as np
import pickle
import os

class NeuralNetwork:
    """
    A neural network implementation using only NumPy for handwritten digit recognition.
    Supports multiple hidden layers with configurable activation functions.
    """
    
    def __init__(self, input_size=784, hidden_sizes=[128, 64], output_size=10, 
                 learning_rate=0.001, activation='relu'):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features (784 for 28x28 MNIST images)
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output classes (10 for digits 0-9)
            learning_rate: Learning rate for gradient descent
            activation: Activation function ('relu', 'sigmoid', 'tanh')
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Create layer sizes list
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Initialize weights using Xavier/Glorot initialization
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization for better convergence
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i + 1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
        
        # Store activations and z values for backpropagation
        self.activations = []
        self.z_values = []
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def _activation_function(self, z, derivative=False):
        """Apply activation function or its derivative."""
        if self.activation == 'relu':
            if derivative:
                return (z > 0).astype(float)
            return np.maximum(0, z)
        
        elif self.activation == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            if derivative:
                return sigmoid * (1 - sigmoid)
            return sigmoid
        
        elif self.activation == 'tanh':
            if derivative:
                tanh = np.tanh(z)
                return 1 - tanh ** 2
            return np.tanh(z)
        
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
    
    def _softmax(self, z):
        """Apply softmax activation for output layer."""
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            Output probabilities of shape (batch_size, output_size)
        """
        self.activations = [X]
        self.z_values = []
        
        current_input = X
        
        # Forward through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            activation = self._activation_function(z)
            self.activations.append(activation)
            current_input = activation
        
        # Output layer with softmax
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_output)
        
        output = self._softmax(z_output)
        self.activations.append(output)
        
        return output
    
    def backward(self, X, y, output):
        """
        Backward propagation to compute gradients.
        
        Args:
            X: Input data
            y: True labels (one-hot encoded)
            output: Network output from forward pass
        """
        m = X.shape[0]  # batch size
        
        # Initialize gradients
        weight_gradients = []
        bias_gradients = []
        
        # Output layer error (softmax + cross-entropy derivative)
        delta = output - y
        
        # Compute gradients for output layer
        weight_grad = np.dot(self.activations[-2].T, delta) / m
        bias_grad = np.mean(delta, axis=0, keepdims=True)
        
        weight_gradients.insert(0, weight_grad)
        bias_gradients.insert(0, bias_grad)
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            # Compute delta for current layer
            delta = np.dot(delta, self.weights[i + 1].T) * self._activation_function(
                self.z_values[i], derivative=True)
            
            # Compute gradients
            weight_grad = np.dot(self.activations[i].T, delta) / m
            bias_grad = np.mean(delta, axis=0, keepdims=True)
            
            weight_gradients.insert(0, weight_grad)
            bias_gradients.insert(0, bias_grad)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def compute_loss(self, y_true, y_pred):
        """Compute cross-entropy loss."""
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Cross-entropy loss
        loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
        return loss
    
    def compute_accuracy(self, y_true, y_pred):
        """Compute accuracy."""
        predicted_classes = np.argmax(y_pred, axis=1)
        true_classes = np.argmax(y_true, axis=1)
        return np.mean(predicted_classes == true_classes)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, 
              batch_size=32, verbose=True):
        """
        Train the neural network.
        
        Args:
            X_train: Training data
            y_train: Training labels (one-hot encoded)
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            verbose: Whether to print training progress
        """
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_loss = 0
            epoch_accuracy = 0
            
            # Mini-batch gradient descent
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward and backward pass
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)
                
                # Accumulate metrics
                batch_loss = self.compute_loss(y_batch, output)
                batch_accuracy = self.compute_accuracy(y_batch, output)
                
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
            
            # Average metrics over batches
            avg_train_loss = epoch_loss / n_batches
            avg_train_accuracy = epoch_accuracy / n_batches
            
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(avg_train_accuracy)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_output = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_output)
                val_accuracy = self.compute_accuracy(y_val, val_output)
                
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}")
    
    def predict(self, X):
        """Make predictions on new data."""
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        return self.forward(X)
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'activation': self.activation,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.input_size = model_data['input_size']
        self.hidden_sizes = model_data['hidden_sizes']
        self.output_size = model_data['output_size']
        self.learning_rate = model_data['learning_rate']
        self.activation = model_data['activation']
        self.train_losses = model_data.get('train_losses', [])
        self.train_accuracies = model_data.get('train_accuracies', [])
        self.val_losses = model_data.get('val_losses', [])
        self.val_accuracies = model_data.get('val_accuracies', [])
        
        print(f"Model loaded from {filepath}")
