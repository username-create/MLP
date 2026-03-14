#include "NeuralNetwork.hpp"

#include <cmath>
#include <cstdlib>
#include <vector>
#include <stdexcept>

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoidDerivative(double y) {
    return y * (1 - y);
}

double NeuralNetwork::getRandom() const {
    return (double)rand() / RAND_MAX * 2 - 1;
}

NeuralNetwork::NeuralNetwork(std::vector<int> t, double lr) 
    : topology(t), learningRate(lr) {
    if (t.size() < 2) {
        throw std::invalid_argument("The network architecture needs at least two layers.");
    }
    for (int size : t) {
        if (size <= 0) {
            throw std::invalid_argument("The number of neurons in each layer must be greater than 0.");
        }
    }

    activations.resize(t.size());
    for (size_t i = 0; i < t.size(); i++) {
        activations[i].resize(t[i], 0);
    }

    for (size_t i = 0; i < t.size() - 1; i++) {
        int currentLayerNodes = t[i];
        int nextLayerNodes = t[i + 1];

        std::vector<std::vector<double>> layerWeights;
        
        for (int j = 0; j < currentLayerNodes; j++) {
            std::vector<double> nodeWeights;
            for (int k = 0; k < nextLayerNodes; k++) {
                nodeWeights.push_back(getRandom());
            }
            layerWeights.push_back(nodeWeights);
        }

        weights.push_back(layerWeights);
        
        std::vector<double> layerBiases;

        for (int k = 0; k < nextLayerNodes; k++) {
            layerBiases.push_back(getRandom());
        }

        biases.push_back(layerBiases);
    }
}

double NeuralNetwork::predict(const std::vector<double>& inputs) {
    if (inputs.size() != topology[0]) {
        throw std::invalid_argument("Input size mismatch!");
    }

    activations[0] = inputs;

    for (size_t i = 0; i < weights.size(); i++) {
        for (size_t j = 0; j < topology[i + 1]; j++) {
            double sum = biases[i][j];

            for (size_t k = 0; k < topology[i]; k++) {
                sum += activations[i][k] * weights[i][k][j];
            }

            activations[i + 1][j] = sigmoid(sum);
        }
    }

    return activations.back()[0];
}

void NeuralNetwork::train(const std::vector<double>& inputs, double target) {
    predict(inputs);

    std::vector<std::vector<double>> gradients;

    gradients.resize(topology.size());
    for (size_t i = 0; i < topology.size(); i++) {
        gradients[i].resize(topology[i]);
    }

    int lastLayer = topology.size() - 1;
    double final_output = activations[lastLayer][0];
    
    gradients[lastLayer][0] = (target - final_output) * sigmoidDerivative(final_output);

    for (int i = lastLayer - 1; i > 0; i--) {
        for (int j = 0; j < topology[i]; j++) {
            double error = 0;

            for (int k = 0; k < topology[i + 1]; k++) {
                error += gradients[i + 1][k] * weights[i][j][k];
            }

            gradients[i][j] = error * sigmoidDerivative(activations[i][j]);
        }
    }

    for (size_t i = 0; i < weights.size(); i++) {
        for (size_t k = 0; k < topology[i + 1]; k++) {     
            biases[i][k] += learningRate * gradients[i + 1][k];

            for (size_t j = 0; j < topology[i]; j++) {
                weights[i][j][k] += learningRate * gradients[i + 1][k] * activations[i][j];
            }
        }
    }
}