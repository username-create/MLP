MLP - Multi-Layer Perceptron Implementation in C++
This project is a custom implementation of a Multi-Layer Perceptron (MLP) neural network in C++. It features a flexible architecture, backpropagation training, and sigmoid activation, specifically tested on the XOR logic gate problem.

Architecture & Topology
The network supports dynamic topology. In the current main.cpp demonstration, the network is configured as:

```cpp
// { Input, [Hidden Layers...], Output }
NeuralNetwork nn({2, 8, 8, 1}, 0.1);
```

Inputs: 2 (for XOR logic)

Hidden Layers: 2 layers with 8 neurons each

Output: 1

Learning Rate: 0.1

Features
Backpropagation: Optimized weight adjustment through gradient descent.

Header-Only Focused: Core logic is clearly separated into NeuralNetwork.hpp and .cpp.

Educational Purpose: Written for clarity to demonstrate how neural networks function at a low level.

Sample Output
After training for 50,000+ epochs, the network successfully achieves the following results:

```text
Input: (0, 0) | Target: 0 | Predict: 0.0012 -> [0]
Input: (0, 1) | Target: 1 | Predict: 0.9985 -> [1]
Input: (1, 0) | Target: 1 | Predict: 0.9982 -> [1]
Input: (1, 1) | Target: 0 | Predict: 0.0021 -> [0]
```

How to Run
Requirement: A C++ compiler (g++, clang, or MSVC).

```bash
g++ main.cpp NeuralNetwork.cpp -o mlp
./mlp
```

License
This project is licensed under the MIT License.