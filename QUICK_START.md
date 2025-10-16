# Quick Start Guide

## Getting Started

1. **Run the main application:**
   ```bash
   python main.py
   ```

2. **Or train directly:**
   ```bash
   python training/train.py
   ```

3. **Or test a trained model:**
   ```bash
   python testing/test.py
   ```

## Step-by-Step First Use

### Option 1: Using the Interactive Menu
1. Run `python main.py`
2. Choose option 1 to download MNIST data
3. Choose option 2 to train the neural network
4. Choose option 3 to evaluate the trained model

### Option 2: Direct Training
1. Run `python training/train.py`
   - This will automatically download data if not present
   - Train the model with default settings
   - Save the trained model
   - Show training progress and final results

### Option 3: Quick Demo
1. Run `python demo.py`
   - Tests all functionality with dummy data
   - No download required
   - Fast execution (~30 seconds)

## What to Expect

### Training Time
- Full training: 2-5 minutes
- Demo: 30 seconds
- Data download: 1-2 minutes (first time only)

### Expected Accuracy
- Training accuracy: 98%+
- Validation accuracy: 97%+
- Test accuracy: 95%+

### Files Created
- `data/`: MNIST dataset files
- `model/`: Trained neural network models
- `model/evaluation/`: Evaluation plots and reports

## Troubleshooting

### If download fails:
- Check internet connection
- Try running `python initializers/data_loader.py` directly

### If training is slow:
- Reduce epochs or batch size in training script
- Use smaller network architecture

### If you get import errors:
- Make sure you're in the project directory
- Install requirements: `pip install -r requirements.txt`