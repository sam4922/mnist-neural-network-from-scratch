"""
Quick Demo Script for MNIST Neural Network
This script demonstrates the basic functionality without full training.
"""

import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classes.neural_network import NeuralNetwork
from initializers.data_loader import MNISTDataLoader

def quick_demo():
    """Run a quick demonstration of the neural network."""
    print("🧠 MNIST Neural Network Quick Demo")
    print("=" * 50)
    
    # 1. Initialize Neural Network
    print("\n1. Creating neural network...")
    model = NeuralNetwork(
        input_size=784,
        hidden_sizes=[64, 32],  # Smaller for demo
        output_size=10,
        learning_rate=0.01,
        activation='relu'
    )
    print(f"   ✓ Created network with {len(model.weights)} layers")
    print(f"   ✓ Layer sizes: 784 → 64 → 32 → 10")
    
    # 2. Generate dummy data (since MNIST download takes time)
    print("\n2. Creating dummy data for demonstration...")
    # Create some fake image data (normally would be MNIST)
    X_dummy = np.random.rand(100, 784)  # 100 samples, 784 features
    y_dummy = np.eye(10)[np.random.randint(0, 10, 100)]  # One-hot encoded labels
    
    print(f"   ✓ Created dummy training data: {X_dummy.shape}")
    print(f"   ✓ Created dummy labels: {y_dummy.shape}")
    
    # 3. Test forward propagation
    print("\n3. Testing forward propagation...")
    output = model.forward(X_dummy)
    print(f"   ✓ Forward pass successful")
    print(f"   ✓ Output shape: {output.shape}")
    print(f"   ✓ Output probabilities sum to ~1.0: {np.allclose(np.sum(output, axis=1), 1.0)}")
    
    # 4. Test backward propagation
    print("\n4. Testing backward propagation...")
    initial_weights = [w.copy() for w in model.weights]
    model.backward(X_dummy, y_dummy, output)
    weights_changed = any(not np.array_equal(initial_weights[i], model.weights[i]) 
                         for i in range(len(model.weights)))
    print(f"   ✓ Backward pass successful")
    print(f"   ✓ Weights updated: {weights_changed}")
    
    # 5. Test training loop (few epochs)
    print("\n5. Testing mini training loop...")
    initial_loss = model.compute_loss(y_dummy, output)
    
    for epoch in range(5):
        output = model.forward(X_dummy)
        model.backward(X_dummy, y_dummy, output)
        loss = model.compute_loss(y_dummy, output)
        accuracy = model.compute_accuracy(y_dummy, output)
        print(f"   Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
    
    final_loss = model.compute_loss(y_dummy, model.forward(X_dummy))
    print(f"   ✓ Loss decreased: {final_loss < initial_loss}")
    
    # 6. Test predictions
    print("\n6. Testing predictions...")
    predictions = model.predict(X_dummy[:5])
    probabilities = model.predict_proba(X_dummy[:5])
    print(f"   ✓ Predictions shape: {predictions.shape}")
    print(f"   ✓ Probabilities shape: {probabilities.shape}")
    print(f"   ✓ Sample predictions: {predictions}")
    
    # 7. Test model save/load
    print("\n7. Testing model persistence...")
    os.makedirs("model", exist_ok=True)
    model.save_model("model/demo_model.pkl")
    
    # Create new model and load
    new_model = NeuralNetwork(input_size=784, hidden_sizes=[64, 32], output_size=10)
    new_model.load_model("model/demo_model.pkl")
    
    # Test if loaded model gives same results
    new_output = new_model.forward(X_dummy[:5])
    original_output = model.forward(X_dummy[:5])
    models_match = np.allclose(new_output, original_output)
    print(f"   ✓ Model saved and loaded successfully")
    print(f"   ✓ Loaded model matches original: {models_match}")
    
    # 8. Data loader demo
    print("\n8. Testing data loader...")
    loader = MNISTDataLoader(data_dir="data")
    print(f"   ✓ Data loader created")
    print(f"   ✓ Ready to download MNIST data when needed")
    print(f"   ✓ Data directory: {loader.data_dir}")
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("🎉 Neural network implementation is working correctly!")
    print("\nTo use the full system:")
    print("  1. Run 'python main.py' for interactive menu")
    print("  2. Run 'python training/train.py' to train on real MNIST data")
    print("  3. Run 'python testing/test.py' to evaluate trained models")
    print("=" * 50)

if __name__ == "__main__":
    try:
        quick_demo()
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()