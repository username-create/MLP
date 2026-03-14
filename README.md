# Simple Neural Network in C++

![Neural Network Graph](https://raw.githubusercontent.com/martinandert/neural-network-graph/master/example.png)

A lightweight, scalable header-only Neural Network implementation in C++. This project demonstrates a generalized architecture capable of learning non-linear logic gates like **XOR**.

## Features
- **Scalable Topology**: Easily adjust the number of hidden layers and neurons per layer.
- **Generalized Architecture**: Supports any number of inputs and outputs.
- **Backpropagation**: Implemented from scratch using the Sigmoid activation function.
- **Defensive Design**: Includes robust error handling and input validation.

## Project Structure
- `NeuralNetwork.hpp`: Class declaration and structure.
- `NeuralNetwork.cpp`: Implementation of forward propagation and training logic.
- `main.cpp`: Test bench for training the XOR truth table.

## How to Configure the Topology
You can customize the brain's complexity in `main.cpp` by modifying the `topology` vector:

```cpp
// { Input, [Hidden Layers...], Output }
NeuralNetwork nn({2, 8, 8, 1}, 0.1);