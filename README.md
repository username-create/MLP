# MLP - Multi-Layer Perceptron C++ Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, header-only focused C++ implementation of a Multi-Layer Perceptron (MLP). This project demonstrates the core concepts of backpropagation and gradient descent without relying on heavy external libraries.

## 🚀 Features
- **Dynamic Topology**: Configure any number of hidden layers and neurons.
- **Backpropagation**: Optimized weight adjustment using the delta rule.
- **Sigmoid Activation**: Smooth activation function for non-linear logic learning.
- **Random Initialization**: Automatic weight randomization for training.

## 🛠 Project Structure
- `NeuralNetwork.hpp`: Class definitions and network architecture.
- `NeuralNetwork.cpp`: Implementation of feedforward and backpropagation logic.
- `main.cpp`: Entry point demonstrating the XOR logic gate training.

## 📈 Sample Output
After training for 50,000+ epochs, the network successfully learns the XOR logic:

```text
Input: (0, 0) | Target: 0 | Predict: 0.0012 -> [0]
Input: (0, 1) | Target: 1 | Predict: 0.9985 -> [1]
Input: (1, 0) | Target: 1 | Predict: 0.9982 -> [1]
Input: (1, 1) | Target: 0 | Predict: 0.0021 -> [0]