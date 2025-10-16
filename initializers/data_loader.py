import numpy as np
import pandas as pd
import requests
import gzip
import os
from urllib.parse import urljoin
import struct

class MNISTDataLoader:
    """
    Data loader for MNIST handwritten digit dataset.
    Downloads and preprocesses the data for neural network training.
    """
    
    def __init__(self, data_dir="data"):
        """
        Initialize the MNIST data loader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        # Updated to use alternative MNIST source
        self.base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

        
        # MNIST file information
        self.files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz'
        }
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Data storage
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def download_file(self, filename):
        """Download a file from the MNIST dataset."""
        url = urljoin(self.base_url, filename)
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"File {filename} already exists, skipping download.")
            return filepath
        
        print(f"Downloading {filename}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Successfully downloaded {filename}")
            return filepath
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
            return None
    
    def download_mnist_data(self):
        """Download all MNIST dataset files."""
        print("Starting MNIST dataset download...")
        
        for file_type, filename in self.files.items():
            filepath = self.download_file(filename)
            if filepath is None:
                print(f"Failed to download {filename}")
                return False
        
        print("MNIST dataset download completed!")
        return True
    
    def _read_mnist_images(self, filename):
        """Read MNIST image data from IDX file format."""
        filepath = os.path.join(self.data_dir, filename)
        
        with gzip.open(filepath, 'rb') as f:
            # Read header
            magic_number = struct.unpack('>I', f.read(4))[0]
            num_images = struct.unpack('>I', f.read(4))[0]
            num_rows = struct.unpack('>I', f.read(4))[0]
            num_cols = struct.unpack('>I', f.read(4))[0]
            
            # Read image data
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, num_rows * num_cols)
            
        return images
    
    def _read_mnist_labels(self, filename):
        """Read MNIST label data from IDX file format."""
        filepath = os.path.join(self.data_dir, filename)
        
        with gzip.open(filepath, 'rb') as f:
            # Read header
            magic_number = struct.unpack('>I', f.read(4))[0]
            num_labels = struct.unpack('>I', f.read(4))[0]
            
            # Read label data
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            
        return labels
    
    def load_data(self, normalize=True, one_hot_encode=True):
        """
        Load and preprocess MNIST data.
        
        Args:
            normalize: Whether to normalize pixel values to [0, 1]
            one_hot_encode: Whether to one-hot encode labels
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        # Download data if not exists
        if not all(os.path.exists(os.path.join(self.data_dir, filename)) 
                  for filename in self.files.values()):
            success = self.download_mnist_data()
            if not success:
                raise RuntimeError("Failed to download MNIST data")
        
        print("Loading MNIST data...")
        
        # Load training data
        self.X_train = self._read_mnist_images(self.files['train_images'])
        self.y_train = self._read_mnist_labels(self.files['train_labels'])
        
        # Load test data
        self.X_test = self._read_mnist_images(self.files['test_images'])
        self.y_test = self._read_mnist_labels(self.files['test_labels'])
        
        # Normalize pixel values
        if normalize:
            self.X_train = self.X_train.astype(np.float32) / 255.0
            self.X_test = self.X_test.astype(np.float32) / 255.0
        
        # One-hot encode labels
        if one_hot_encode:
            self.y_train = self._one_hot_encode(self.y_train)
            self.y_test = self._one_hot_encode(self.y_test)
        
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Training labels shape: {self.y_train.shape}")
        print(f"Test data shape: {self.X_test.shape}")
        print(f"Test labels shape: {self.y_test.shape}")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def _one_hot_encode(self, labels, num_classes=10):
        """Convert labels to one-hot encoding."""
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot
    
    def create_validation_split(self, validation_split=0.1):
        """
        Create validation split from training data.
        
        Args:
            validation_split: Fraction of training data to use for validation
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Calculate split index
        n_samples = self.X_train.shape[0]
        n_val = int(n_samples * validation_split)
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        # Split data
        X_val = self.X_train[val_indices]
        y_val = self.y_train[val_indices]
        X_train_split = self.X_train[train_indices]
        y_train_split = self.y_train[train_indices]
        
        print(f"Training split shape: {X_train_split.shape}")
        print(f"Validation split shape: {X_val.shape}")
        
        return X_train_split, y_train_split, X_val, y_val
    
    def get_data_stats(self):
        """Get statistics about the loaded data."""
        if self.X_train is None:
            print("No data loaded. Call load_data() first.")
            return
        
        stats = {
            'train_samples': self.X_train.shape[0],
            'test_samples': self.X_test.shape[0],
            'image_size': self.X_train.shape[1],
            'pixel_mean': np.mean(self.X_train),
            'pixel_std': np.std(self.X_train),
            'pixel_min': np.min(self.X_train),
            'pixel_max': np.max(self.X_train)
        }
        
        print("MNIST Dataset Statistics:")
        print("-" * 30)
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        return stats
    
    def visualize_samples(self, num_samples=10, save_path=None):
        """
        Visualize random samples from the training data.
        
        Args:
            num_samples: Number of samples to visualize
            save_path: Optional path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Install it to visualize samples.")
            return
        
        if self.X_train is None or self.y_train is None:
            print("No data loaded. Call load_data() first.")
            return
        
        # Select random samples
        indices = np.random.choice(len(self.X_train), num_samples, replace=False)
        
        # Create subplot grid
        cols = min(5, num_samples)
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            row = i // cols
            col = i % cols
            
            # Reshape image to 28x28
            image = self.X_train[idx].reshape(28, 28)
            
            # Get label (convert from one-hot if needed)
            if len(self.y_train.shape) > 1:
                label = np.argmax(self.y_train[idx])
            else:
                label = self.y_train[idx]
            
            axes[row, col].imshow(image, cmap='gray')
            axes[row, col].set_title(f'Label: {label}')
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def save_processed_data(self, filepath):
        """Save processed data to disk."""
        if self.X_train is None:
            print("No data to save. Load data first.")
            return
        
        data = {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_test': self.X_test,
            'y_test': self.y_test
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez_compressed(filepath, **data)
        print(f"Processed data saved to {filepath}")
    
    def load_processed_data(self, filepath):
        """Load processed data from disk."""
        if not os.path.exists(filepath):
            print(f"File {filepath} not found.")
            return False
        
        data = np.load(filepath)
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        
        print(f"Processed data loaded from {filepath}")
        return True


def main():
    """Example usage of the MNIST data loader."""
    # Initialize data loader
    loader = MNISTDataLoader()
    
    # Load and preprocess data
    X_train, y_train, X_test, y_test = loader.load_data()
    
    # Get data statistics
    loader.get_data_stats()
    
    # Create validation split
    X_train_split, y_train_split, X_val, y_val = loader.create_validation_split()
    
    # Visualize some samples
    loader.visualize_samples(num_samples=10)
    
    # Save processed data
    loader.save_processed_data("data/mnist_processed.npz")


if __name__ == "__main__":
    main()