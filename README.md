# FCNN MNIST Project

This repository contains a simple implementation of a Fully Connected Neural Network (FCNN) for classifying the MNIST dataset. The goal of this project is to provide a clear and concise example of how to build, train, and evaluate a neural network for image classification using Python and PyTorch.

## File Descriptions

- `main.py`: The main script to run the training and testing of the model.
- `model.py`: Defines the architecture of the Fully Connected Neural Network.
- `train_loop.py`: Contains the function for the training loop, which iterates over the training data, computes the loss, and updates the model parameters.
- `test_loop.py`: Contains the function for the testing loop, which evaluates the model's performance on the test data.
- `visualize.py`: A script to generate a plot of the training and testing metrics, which are saved in `train_metrics.csv` and `test_metrics.csv`.
- `mnist_metrics.png`: The output image from `visualize.py`.

## How to Run

1. **Train and evaluate the model:**
   ```bash
   python src/main.py
   ```

2. **Visualize the metrics:**
   ```bash
   python src/visualize.py
   ```

## Visualization

![MNIST Metrics](mnist_metrics.png)
