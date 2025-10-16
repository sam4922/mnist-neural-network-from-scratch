import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classes.neural_network import NeuralNetwork
from initializers.data_loader import MNISTDataLoader

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix."""
    # Convert one-hot to class labels if needed
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_misclassified_samples(X_test, y_true, y_pred, num_samples=20, save_path=None):
    """Plot misclassified samples."""
    # Convert one-hot to class labels if needed
    if len(y_true.shape) > 1:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true
    
    if len(y_pred.shape) > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_pred_labels = y_pred
    
    # Find misclassified indices
    misclassified_indices = np.where(y_true_labels != y_pred_labels)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassified samples found!")
        return
    
    # Select random misclassified samples
    num_samples = min(num_samples, len(misclassified_indices))
    selected_indices = np.random.choice(misclassified_indices, num_samples, replace=False)
    
    # Create subplot grid
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(selected_indices):
        row = i // cols
        col = i % cols
        
        # Reshape image to 28x28
        image = X_test[idx].reshape(28, 28)
        
        true_label = y_true_labels[idx]
        pred_label = y_pred_labels[idx]
        
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(f'True: {true_label}, Pred: {pred_label}', 
                                fontsize=10, color='red')
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle('Misclassified Samples', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Misclassified samples plot saved to {save_path}")
    
    plt.show()

def plot_prediction_confidence(y_true, y_pred_proba, save_path=None):
    """Plot prediction confidence distribution."""
    # Get maximum confidence for each prediction
    max_confidences = np.max(y_pred_proba, axis=1)
    
    # Convert one-hot to class labels if needed
    if len(y_true.shape) > 1:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true
    
    y_pred_labels = np.argmax(y_pred_proba, axis=1)
    
    # Separate correct and incorrect predictions
    correct_mask = y_true_labels == y_pred_labels
    correct_confidences = max_confidences[correct_mask]
    incorrect_confidences = max_confidences[~correct_mask]
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(correct_confidences, bins=50, alpha=0.7, label='Correct Predictions', 
             color='green', density=True)
    plt.hist(incorrect_confidences, bins=50, alpha=0.7, label='Incorrect Predictions', 
             color='red', density=True)
    
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Density')
    plt.title('Distribution of Prediction Confidence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confidence distribution plot saved to {save_path}")
    
    plt.show()

def analyze_per_class_performance(y_true, y_pred, y_pred_proba):
    """Analyze performance for each digit class."""
    # Convert one-hot to class labels if needed
    if len(y_true.shape) > 1:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true
    
    if len(y_pred.shape) > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_pred_labels = y_pred
    
    print("\nPer-Class Performance Analysis:")
    print("=" * 50)
    print(f"{'Digit':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 50)
    
    # Calculate metrics for each class
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average=None)
    
    for digit in range(10):
        # Calculate accuracy for this digit
        digit_mask = y_true_labels == digit
        digit_accuracy = np.mean(y_pred_labels[digit_mask] == digit) if np.any(digit_mask) else 0
        
        print(f"{digit:<6} {digit_accuracy:<10.4f} {precision[digit]:<10.4f} "
              f"{recall[digit]:<10.4f} {f1[digit]:<10.4f}")
    
    # Overall metrics
    overall_accuracy = np.mean(y_true_labels == y_pred_labels)
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)
    
    print("-" * 50)
    print(f"{'Avg':<6} {overall_accuracy:<10.4f} {avg_precision:<10.4f} "
          f"{avg_recall:<10.4f} {avg_f1:<10.4f}")
    
    return precision, recall, f1, support

def test_model_interactive():
    """Interactive model testing with user input."""
    print("Interactive Model Testing")
    print("=" * 30)
    
    # Load model
    model_path = "model/mnist_neural_network.pkl"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        print("Please train a model first using training/train.py")
        return
    
    model = NeuralNetwork(input_size=784, hidden_sizes=[256, 128], output_size=10)
    model.load_model(model_path)
    
    # Load test data
    loader = MNISTDataLoader(data_dir="data")
    X_train, y_train, X_test, y_test = loader.load_data()
    
    while True:
        print("\nOptions:")
        print("1. Test on random sample")
        print("2. Test on specific index")
        print("3. Show prediction probabilities")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            # Random sample
            idx = np.random.randint(0, len(X_test))
            sample = X_test[idx:idx+1]
            true_label = np.argmax(y_test[idx]) if len(y_test.shape) > 1 else y_test[idx]
            
            prediction = model.predict(sample)[0]
            probabilities = model.predict_proba(sample)[0]
            
            print(f"\nRandom sample #{idx}")
            print(f"True label: {true_label}")
            print(f"Predicted label: {prediction}")
            print(f"Confidence: {probabilities[prediction]:.4f}")
            
            # Show image
            plt.figure(figsize=(4, 4))
            plt.imshow(sample.reshape(28, 28), cmap='gray')
            plt.title(f'True: {true_label}, Predicted: {prediction}')
            plt.axis('off')
            plt.show()
            
        elif choice == "2":
            # Specific index
            try:
                idx = int(input(f"Enter index (0-{len(X_test)-1}): "))
                if 0 <= idx < len(X_test):
                    sample = X_test[idx:idx+1]
                    true_label = np.argmax(y_test[idx]) if len(y_test.shape) > 1 else y_test[idx]
                    
                    prediction = model.predict(sample)[0]
                    probabilities = model.predict_proba(sample)[0]
                    
                    print(f"\nSample #{idx}")
                    print(f"True label: {true_label}")
                    print(f"Predicted label: {prediction}")
                    print(f"Confidence: {probabilities[prediction]:.4f}")
                    
                    # Show image
                    plt.figure(figsize=(4, 4))
                    plt.imshow(sample.reshape(28, 28), cmap='gray')
                    plt.title(f'True: {true_label}, Predicted: {prediction}')
                    plt.axis('off')
                    plt.show()
                else:
                    print("Invalid index!")
            except ValueError:
                print("Please enter a valid number!")
                
        elif choice == "3":
            # Show probabilities
            try:
                idx = int(input(f"Enter index (0-{len(X_test)-1}): "))
                if 0 <= idx < len(X_test):
                    sample = X_test[idx:idx+1]
                    true_label = np.argmax(y_test[idx]) if len(y_test.shape) > 1 else y_test[idx]
                    
                    probabilities = model.predict_proba(sample)[0]
                    
                    print(f"\nPrediction probabilities for sample #{idx}:")
                    print(f"True label: {true_label}")
                    print("-" * 25)
                    for digit in range(10):
                        marker = " <-- TRUE" if digit == true_label else ""
                        print(f"Digit {digit}: {probabilities[digit]:.4f}{marker}")
                    
                    # Plot probabilities
                    plt.figure(figsize=(10, 4))
                    
                    plt.subplot(1, 2, 1)
                    plt.imshow(sample.reshape(28, 28), cmap='gray')
                    plt.title(f'Sample #{idx} (True: {true_label})')
                    plt.axis('off')
                    
                    plt.subplot(1, 2, 2)
                    colors = ['red' if i == true_label else 'blue' for i in range(10)]
                    plt.bar(range(10), probabilities, color=colors, alpha=0.7)
                    plt.xlabel('Digit')
                    plt.ylabel('Probability')
                    plt.title('Prediction Probabilities')
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.show()
                else:
                    print("Invalid index!")
            except ValueError:
                print("Please enter a valid number!")
                
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")

def comprehensive_model_evaluation():
    """Comprehensive evaluation of the trained model."""
    print("Comprehensive Model Evaluation")
    print("=" * 40)
    
    # Load model
    model_path = "model/mnist_neural_network.pkl"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        print("Please train a model first using training/train.py")
        return
    
    print("1. Loading trained model...")
    model = NeuralNetwork(input_size=784, hidden_sizes=[256, 128], output_size=10)
    model.load_model(model_path)
    
    # Load test data
    print("2. Loading test data...")
    loader = MNISTDataLoader(data_dir="data")
    X_train, y_train, X_test, y_test = loader.load_data()
    
    # Make predictions
    print("3. Making predictions on test set...")
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    print("4. Calculating metrics...")
    test_accuracy = model.compute_accuracy(y_test, y_pred_proba)
    test_loss = model.compute_loss(y_test, y_pred_proba)
    
    print(f"\nOverall Test Performance:")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Per-class analysis
    print("\n5. Analyzing per-class performance...")
    precision, recall, f1, support = analyze_per_class_performance(y_test, y_pred, y_pred_proba)
    
    # Confusion matrix
    print("\n6. Generating confusion matrix...")
    os.makedirs("model/evaluation", exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, save_path="model/evaluation/confusion_matrix.png")
    
    # Misclassified samples
    print("\n7. Analyzing misclassified samples...")
    plot_misclassified_samples(X_test, y_test, y_pred, 
                              save_path="model/evaluation/misclassified_samples.png")
    
    # Prediction confidence
    print("\n8. Analyzing prediction confidence...")
    plot_prediction_confidence(y_test, y_pred_proba, 
                              save_path="model/evaluation/confidence_distribution.png")
    
    # Classification report
    print("\n9. Generating detailed classification report...")
    y_true_labels = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    y_pred_labels = np.argmax(y_pred_proba, axis=1) if len(y_pred_proba.shape) > 1 else y_pred
    
    report = classification_report(y_true_labels, y_pred_labels, 
                                 target_names=[f'Digit {i}' for i in range(10)])
    print("\nDetailed Classification Report:")
    print(report)
    
    # Save report to file
    with open("model/evaluation/classification_report.txt", "w") as f:
        f.write("MNIST Neural Network Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(report)
    
    print("\nEvaluation complete! Results saved to model/evaluation/")

if __name__ == "__main__":
    print("MNIST Neural Network Testing")
    print("1. Comprehensive model evaluation")
    print("2. Interactive testing")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        comprehensive_model_evaluation()
    elif choice == "2":
        test_model_interactive()
    else:
        print("Invalid choice. Running comprehensive evaluation...")
        comprehensive_model_evaluation()