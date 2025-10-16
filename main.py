#!/usr/bin/env python3
"""
MNIST Neural Network for Handwritten Digit Recognition
Built from scratch using only NumPy/Pandas (no pre-built ML frameworks)

This is the main entry point for the MNIST digit recognition project.
"""

import sys
import os

def print_banner():
    """Print project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        MNIST Neural Network Digit Recognition System         ║
    ║                 (Built from Scratch - No Frameworks)        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main entry point for the application."""
    print_banner()
    
    print("Welcome to the MNIST Neural Network Project!")
    print("\nThis project implements a neural network from scratch using only NumPy")
    print("for handwritten digit recognition on the MNIST dataset.")
    
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Download and prepare MNIST data")
        print("2. Train neural network model")
        print("3. Test/evaluate trained model")
        print("4. Interactive model testing")
        print("5. Train multiple model configurations")
        print("6. View project information")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            print("\nDownloading and preparing MNIST data...")
            try:
                from initializers.data_loader import MNISTDataLoader
                loader = MNISTDataLoader(data_dir="data")
                X_train, y_train, X_test, y_test = loader.load_data()
                loader.get_data_stats()
                print("\nData preparation completed successfully!")
                
                # Optionally visualize samples
                visualize = input("\nWould you like to visualize some samples? (y/n): ").strip().lower()
                if visualize == 'y':
                    loader.visualize_samples(num_samples=10)
                    
            except Exception as e:
                print(f"Error preparing data: {e}")
        
        elif choice == "2":
            print("\nStarting neural network training...")
            try:
                from training.train import train_model
                train_model()
            except Exception as e:
                print(f"Error during training: {e}")
        
        elif choice == "3":
            print("\nEvaluating trained model...")
            try:
                from testing.test import comprehensive_model_evaluation
                comprehensive_model_evaluation()
            except Exception as e:
                print(f"Error during evaluation: {e}")
        
        elif choice == "4":
            print("\nStarting interactive model testing...")
            try:
                from testing.test import test_model_interactive
                test_model_interactive()
            except Exception as e:
                print(f"Error during interactive testing: {e}")
        
        elif choice == "5":
            print("\nTraining multiple model configurations...")
            try:
                from training.train import train_with_different_configurations
                train_with_different_configurations()
            except Exception as e:
                print(f"Error during multi-config training: {e}")
        
        elif choice == "6":
            show_project_info()
        
        elif choice == "7":
            print("\nThank you for using the MNIST Neural Network Project!")
            print("Goodbye! 👋")
            break
        
        else:
            print("Invalid choice! Please enter a number between 1-7.")

def show_project_info():
    """Display project information."""
    info = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                      PROJECT INFORMATION                     ║
    ╚══════════════════════════════════════════════════════════════╝
    
    """
    print(info)

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'requests', 
        'PIL', 'seaborn', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

if __name__ == "__main__":
    # Check dependencies before starting
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye! 👋")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check your installation and try again.")