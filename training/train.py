import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classes.neural_network import NeuralNetwork
from initializers.data_loader import MNISTDataLoader

def plot_training_history(model, save_path=None):
    """Plot training history (loss and accuracy)."""
    epochs = range(1, len(model.train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(epochs, model.train_losses, 'b-', label='Training Loss')
    if model.val_losses:
        ax1.plot(epochs, model.val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, model.train_accuracies, 'b-', label='Training Accuracy')
    if model.val_accuracies:
        ax2.plot(epochs, model.val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()

def train_model():
    """Main training function."""
    print("Starting Neural Network Training for MNIST Digit Recognition")
    print("=" * 60)
    
    # Initialize data loader
    print("1. Loading MNIST dataset...")
    loader = MNISTDataLoader(data_dir="data")
    
    # Load data
    X_train, y_train, X_test, y_test = loader.load_data(normalize=True, one_hot_encode=True)
    
    # Create validation split
    X_train_split, y_train_split, X_val, y_val = loader.create_validation_split(validation_split=0.1)
    
    # Display data information
    print(f"\nDataset Information:")
    print(f"Training samples: {X_train_split.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Input features: {X_train_split.shape[1]}")
    
    # Initialize neural network
    print("\n2. Initializing Neural Network...")
    
    # Network configuration
    config = {
        'input_size': 784,           # 28x28 images
        'hidden_sizes': [256, 128],  # Two hidden layers
        'output_size': 10,           # 10 digit classes
        'learning_rate': 0.001,      # Learning rate
        'activation': 'relu'         # Activation function
    }
    
    print(f"Network Architecture:")
    print(f"  Input Layer: {config['input_size']} neurons")
    for i, size in enumerate(config['hidden_sizes']):
        print(f"  Hidden Layer {i+1}: {size} neurons ({config['activation']} activation)")
    print(f"  Output Layer: {config['output_size']} neurons (softmax activation)")
    print(f"  Learning Rate: {config['learning_rate']}")
    
    # Create neural network
    model = NeuralNetwork(**config)
    
    # Training parameters
    training_params = {
        'epochs': 100,
        'batch_size': 128,
        'verbose': True
    }
    
    print(f"\nTraining Parameters:")
    print(f"  Epochs: {training_params['epochs']}")
    print(f"  Batch Size: {training_params['batch_size']}")
    
    # Train the model
    print("\n3. Training Neural Network...")
    print("-" * 40)
    
    start_time = time.time()
    
    model.train(
        X_train_split, y_train_split,
        X_val, y_val,
        **training_params
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("\n4. Evaluating on Test Set...")
    test_predictions = model.predict_proba(X_test)
    test_accuracy = model.compute_accuracy(y_test, test_predictions)
    test_loss = model.compute_loss(y_test, test_predictions)
    
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save the trained model
    print("\n5. Saving Model...")
    os.makedirs("model", exist_ok=True)
    model_path = "model/mnist_neural_network.pkl"
    model.save_model(model_path)
    
    # Plot training history
    print("\n6. Plotting Training History...")
    plot_training_history(model, save_path="model/training_history.png")
    
    # Display final results
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Final Training Accuracy: {model.train_accuracies[-1]:.4f} ({model.train_accuracies[-1]*100:.2f}%)")
    print(f"Final Validation Accuracy: {model.val_accuracies[-1]:.4f} ({model.val_accuracies[-1]*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Model saved to: {model_path}")
    print("=" * 60)
    
    return model, test_accuracy

def train_with_different_configurations():
    """Train models with different configurations and compare results."""
    print("Training Multiple Model Configurations")
    print("=" * 50)
    
    # Different configurations to try
    configurations = [
        {
            'name': 'Small Network',
            'hidden_sizes': [64],
            'learning_rate': 0.001,
            'activation': 'relu'
        },
        {
            'name': 'Medium Network',
            'hidden_sizes': [128, 64],
            'learning_rate': 0.001,
            'activation': 'relu'
        },
        {
            'name': 'Large Network',
            'hidden_sizes': [256, 128, 64],
            'learning_rate': 0.001,
            'activation': 'relu'
        },
        {
            'name': 'Sigmoid Activation',
            'hidden_sizes': [128, 64],
            'learning_rate': 0.01,
            'activation': 'sigmoid'
        }
    ]
    
    # Load data once
    loader = MNISTDataLoader(data_dir="data")
    X_train, y_train, X_test, y_test = loader.load_data()
    X_train_split, y_train_split, X_val, y_val = loader.create_validation_split(0.1)
    
    results = []
    
    for i, config in enumerate(configurations):
        print(f"\n{i+1}. Training {config['name']}...")
        print(f"   Architecture: {config['hidden_sizes']}")
        print(f"   Activation: {config['activation']}")
        print(f"   Learning Rate: {config['learning_rate']}")
        
        # Create and train model
        model = NeuralNetwork(
            input_size=784,
            hidden_sizes=config['hidden_sizes'],
            output_size=10,
            learning_rate=config['learning_rate'],
            activation=config['activation']
        )
        
        start_time = time.time()
        model.train(X_train_split, y_train_split, X_val, y_val, 
                   epochs=50, batch_size=128, verbose=False)
        training_time = time.time() - start_time
        
        # Evaluate
        test_pred = model.predict_proba(X_test)
        test_accuracy = model.compute_accuracy(y_test, test_pred)
        
        result = {
            'name': config['name'],
            'config': config,
            'test_accuracy': test_accuracy,
            'training_time': training_time,
            'final_train_acc': model.train_accuracies[-1],
            'final_val_acc': model.val_accuracies[-1]
        }
        results.append(result)
        
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        print(f"   Training Time: {training_time:.2f}s")
        
        # Save model
        model_path = f"model/{config['name'].lower().replace(' ', '_')}_model.pkl"
        model.save_model(model_path)
    
    # Print comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model Name':<20} {'Test Acc':<10} {'Train Acc':<10} {'Val Acc':<10} {'Time (s)':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<20} "
              f"{result['test_accuracy']:.4f}    "
              f"{result['final_train_acc']:.4f}     "
              f"{result['final_val_acc']:.4f}      "
              f"{result['training_time']:.1f}")
    
    # Find best model
    best_model = max(results, key=lambda x: x['test_accuracy'])
    print(f"\nBest Model: {best_model['name']} with {best_model['test_accuracy']:.4f} test accuracy")

if __name__ == "__main__":
    # Allow user to choose training mode
    print("MNIST Neural Network Training")
    print("1. Train single model with default configuration")
    print("2. Train and compare multiple configurations")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        train_model()
    elif choice == "2":
        train_with_different_configurations()
    else:
        print("Invalid choice. Training with default configuration...")
        train_model()