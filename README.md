# MLP-MNIST-from-Scratch 🧠

A Multi-Layer Perceptron (MLP) neural network implemented in **pure C++** for handwritten digit recognition. This project avoids high-level frameworks like TensorFlow or PyTorch to focus on the fundamental implementation of deep learning algorithms.

## 🚀 Technical Highlights

* **Pure C++ Implementation**: Built from scratch, including matrix operations, backpropagation, and weight update logic.
* **Hardware Acceleration (OpenMP)**: Utilizes multi-threading to significantly optimize training speed for large datasets (60,000 samples).
* **Model Persistence**: Supports `saveModel` and `loadModel` functionality, allowing for **Warm Starts** by saving trained weights and biases to `.txt` files.
* **Error Analysis System**: Includes a built-in testing mechanism that captures misclassified samples and visualizes them using ASCII art to help analyze model weaknesses.
* **Training Optimization**: Implements per-epoch data shuffling using `std::shuffle` and `mt19937` to prevent overfitting.

## 📊 Performance

* **Test Platform**: Intel Core i7-14650HX, 32GB RAM.
* **Accuracy**:
    * Training Set: ~100%
    * Test Set: **> 95%** (Varies based on topology and training epochs).

## 🛠️ Installation & Usage

### 1. Prepare Data
Ensure `mnist_train.csv` and `mnist_test.csv` are located in the project root directory.

### 2. Compile
```bash
g++ -O3 -fopenmp main.cpp NeuralNetwork.cpp -o mlp_mnist
```

### 3. Run
```bash
./mlp_mnist
```

## 📂 File Structure

* `main.cpp`: Main logic, including data loading, training loops, and error sampling.
* `NeuralNetwork.hpp/cpp`: Neural network class handling all mathematical operations and Model I/O.
* `mnist_model.txt`: Exported weights and biases (generated automatically).

## 🔍 Error Analysis Example
The program visualizes cases where the AI "tripped," allowing for qualitative analysis:
```text
[Error Case] Index: 452
@@@@@@@@@@@@
      @@    
      @@    
    @@@@    
Real: [4] | AI guessed: [9]
```

License
This project is licensed under the MIT License.