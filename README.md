# MNIST Neural Network for Handwritten Digit Recognition

A complete neural network implementation built from scratch using only NumPy/Pandas for handwritten digit recognition on the MNIST dataset. No pre-built machine learning frameworks are used - everything is implemented from the ground up to demonstrate the fundamentals of neural networks.

## 🎯 Project Overview

This project implements a multi-layer neural network with:
- **Configurable Architecture**: Customizable hidden layers, neurons, and activation functions
- **Multiple Activation Functions**: ReLU, Sigmoid, and Tanh
- **Backpropagation**: Full gradient descent implementation
- **Mini-batch Training**: Efficient batch processing
- **Model Persistence**: Save and load trained models
- **Comprehensive Evaluation**: Multiple metrics and visualizations

### Achieved Performance
- **Test Accuracy**: 95%+ on MNIST test set
- **Training Time**: 2-5 minutes (depending on configuration)
- **Model Size**: Compact and efficient

## 📁 Project Structure

```
neural_network_digit/
├── classes/
│   └── neural_network.py      # Core neural network implementation
├── data/                      # MNIST dataset storage (auto-downloaded)
├── initializers/
│   └── data_loader.py         # MNIST data download and preprocessing
├── model/                     # Trained model storage
│   ├── evaluation/            # Evaluation results and plots
│   └── *.pkl                  # Saved model files
├── training/
│   └── train.py               # Model training scripts
├── testing/
│   └── test.py                # Model evaluation and testing
├── main.py                    # Main entry point
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or download the project
cd neural_network_digit

# Install required packages
pip install numpy pandas matplotlib requests pillow scikit-learn seaborn
```

### 2. Run the Project

```bash
python main.py
```

This will launch an interactive menu with the following options:
1. Download and prepare MNIST data
2. Train neural network model
3. Test/evaluate trained model
4. Interactive model testing
5. Train multiple model configurations
6. View project information
7. Exit

### 3. Alternative: Direct Training

```bash
# Train a model directly
python training/train.py

# Test the trained model
python testing/test.py
```

## 🧠 Neural Network Architecture

### Default Configuration
- **Input Layer**: 784 neurons (28×28 pixel images)
- **Hidden Layer 1**: 256 neurons (ReLU activation)
- **Hidden Layer 2**: 128 neurons (ReLU activation)
- **Output Layer**: 10 neurons (Softmax activation for digit classes 0-9)

### Key Features
- **Xavier/Glorot Weight Initialization**: For better convergence
- **Cross-entropy Loss**: Optimal for multi-class classification
- **Mini-batch Gradient Descent**: Efficient training with configurable batch sizes
- **Validation Split**: Monitor overfitting during training

## 📊 MNIST Dataset

The MNIST dataset consists of:
- **Training Set**: 60,000 handwritten digit images
- **Test Set**: 10,000 handwritten digit images
- **Image Size**: 28×28 pixels (grayscale)
- **Classes**: 10 digits (0-9)

The data is automatically downloaded from the official MNIST repository when first running the program.

## 🔧 Configuration Options

### Neural Network Parameters
```python
config = {
    'input_size': 784,           # Fixed for MNIST
    'hidden_sizes': [256, 128],  # Customizable hidden layers
    'output_size': 10,           # Fixed for 10 digit classes
    'learning_rate': 0.001,      # Learning rate
    'activation': 'relu'         # 'relu', 'sigmoid', or 'tanh'
}
```

### Training Parameters
```python
training_params = {
    'epochs': 100,               # Number of training epochs
    'batch_size': 128,           # Mini-batch size
    'validation_split': 0.1      # Fraction for validation
}
```

## 📈 Evaluation Metrics

The project provides comprehensive evaluation including:

### Performance Metrics
- **Accuracy**: Overall and per-class accuracy
- **Precision, Recall, F1-Score**: For each digit class
- **Confusion Matrix**: Visual representation of classification results
- **Loss Curves**: Training and validation loss over time

### Visualizations
- **Training History**: Loss and accuracy plots
- **Confusion Matrix Heatmap**: Class-wise performance
- **Misclassified Samples**: Examples of incorrect predictions
- **Prediction Confidence**: Distribution of model confidence

### Interactive Testing
- Test individual samples
- View prediction probabilities
- Analyze model confidence
- Real-time predictions

## 💡 Implementation Details

### Core Classes and Functions

#### `NeuralNetwork` Class
- `__init__()`: Initialize network architecture
- `forward()`: Forward propagation
- `backward()`: Backpropagation and weight updates
- `train()`: Main training loop with mini-batches
- `predict()`: Make predictions on new data
- `save_model()` / `load_model()`: Model persistence

#### `MNISTDataLoader` Class
- `download_mnist_data()`: Automatic dataset download
- `load_data()`: Data preprocessing and normalization
- `create_validation_split()`: Split training data
- `visualize_samples()`: Data visualization

### Mathematical Implementation

#### Forward Propagation
```
z = X @ W + b
a = activation_function(z)
```

#### Backpropagation
```
δ_output = output - y_true
δ_hidden = δ_next @ W_next.T * activation_derivative(z)
∇W = X.T @ δ / batch_size
∇b = mean(δ, axis=0)
```

#### Weight Updates
```
W = W - learning_rate * ∇W
b = b - learning_rate * ∇b
```

## 🎨 Customization Examples

### Train with Different Architectures

```python
# Small network
model = NeuralNetwork(
    hidden_sizes=[64],
    learning_rate=0.001,
    activation='relu'
)

# Large network
model = NeuralNetwork(
    hidden_sizes=[512, 256, 128],
    learning_rate=0.0005,
    activation='relu'
)

# Sigmoid activation
model = NeuralNetwork(
    hidden_sizes=[128, 64],
    learning_rate=0.01,
    activation='sigmoid'
)
```

### Custom Training Parameters

```python
model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=200,
    batch_size=64,
    verbose=True
)
```

## 📋 Requirements

### Python Version
- Python 3.7 or higher

### Required Packages
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib`: Plotting and visualization
- `requests`: HTTP requests for data download
- `pillow`: Image processing
- `scikit-learn`: Evaluation metrics
- `seaborn`: Statistical visualizations

### Installation
```bash
pip install numpy pandas matplotlib requests pillow scikit-learn seaborn
```

## 🚀 Performance Optimization

### Training Speedup Tips
1. **Batch Size**: Larger batches for faster training (if memory allows)
2. **Learning Rate**: Tune for faster convergence
3. **Architecture**: Smaller networks train faster
4. **Early Stopping**: Monitor validation loss

### Memory Optimization
1. **Gradient Accumulation**: For very large datasets
2. **Data Loading**: Stream data instead of loading all at once
3. **Model Checkpointing**: Save during training

## 🔍 Troubleshooting

### Common Issues

#### Download Problems
- **Solution**: Check internet connection, try running `initializers/data_loader.py` directly

#### Training Not Converging
- **Solutions**: 
  - Lower learning rate
  - Try different activation functions
  - Check data preprocessing
  - Increase network capacity

#### Memory Errors
- **Solutions**:
  - Reduce batch size
  - Use smaller network architecture
  - Close other applications

#### Poor Accuracy
- **Solutions**:
  - Train for more epochs
  - Increase learning rate
  - Add more hidden layers/neurons
  - Check data normalization

## 🎓 Educational Value

This project demonstrates:
- **Neural Network Fundamentals**: Forward/backward propagation
- **Gradient Descent**: Optimization algorithms
- **Activation Functions**: Different function behaviors
- **Loss Functions**: Cross-entropy for classification
- **Regularization**: Through architecture choices
- **Evaluation**: Comprehensive model assessment
- **Software Engineering**: Clean, modular code structure

## 🤝 Contributing

Feel free to:
- Add new activation functions
- Implement different optimizers (Adam, RMSprop)
- Add regularization techniques (dropout, L2)
- Improve visualization features
- Add new evaluation metrics

## 📜 License

This project is created for educational purposes. Feel free to use and modify for learning.

## 🙏 Acknowledgments

- MNIST dataset creators (Yann LeCun et al.)
- The neural network research community
- NumPy and scientific Python ecosystem

---

**Happy Learning! 🎉**

Built with ❤️ for understanding neural networks from the ground up.