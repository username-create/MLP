#include "NeuralNetwork.hpp"

#include <cmath>
#include <random>
#include <vector>
#include <stdexcept>
#include <ctime>
#include <omp.h>
#include <fstream>
#include <iostream>

double ReLU(double x) {
    return x > 0 ? x : 0;
}

double ReLUDerivative(double y) {
    return y > 0 ? 1 : 0;
}

double NeuralNetwork::getRandom() const {
    return (double)rand() / RAND_MAX * 2 - 1;
}

void NeuralNetwork::saveModel(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) return;

    for (auto& layer : weights) {
        for (auto& node : layer) {
            for (double w : node) out << w << " ";
            out << "\n";
        }
    }

    for (auto& layer : biases) {
        for (double b : layer) out << b << " ";
        out << "\n";
    }
    out.close();
    std::cout << "Model saved to " << filename << "\n";
}

void NeuralNetwork::loadModel(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) return;

    for (auto& layer : weights) {
        for (auto& node : layer) {
            for (double& w : node) in >> w;
        }
    }
    for (auto& layer : biases) {
        for (double& b : layer) in >> b;
    }
    in.close();
    std::cout << "Model loaded from " << filename << "\n";
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
        int n_in = t[i];
        double stddev = std::sqrt(2.0 / n_in);

        std::default_random_engine generator(time(NULL) + i);
        std::normal_distribution<double> distribution(0.0, stddev);
        std::vector<std::vector<double>> layerWeights;
        
        for (int j = 0; j < n_in; j++) {
            std::vector<double> nodeWeights;
            for (int k = 0; k < t[i + 1]; k++) {
                nodeWeights.push_back(distribution(generator));
            }
            layerWeights.push_back(nodeWeights);
        }

        weights.push_back(layerWeights);
        
        std::vector<double> layerBiases;

        for (int k = 0; k < t[i + 1]; k++) {
            layerBiases.push_back(0.01); 
        }

        biases.push_back(layerBiases);
    }

    gradients.resize(t.size());
    for (size_t i = 0; i < t.size(); i++) {
        gradients[i].resize(t[i], 0.0);
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& inputs) {
    if (inputs.size() != topology[0]) {
        throw std::invalid_argument("Input size mismatch!");
    }

    activations[0] = inputs;

    for (size_t i = 0; i < weights.size(); i++) {
        bool isOutputLayer = (i == weights.size() - 1);
        int next_size = (int)topology[i + 1];

        #pragma omp parallel for if(next_size > 512)
        for (size_t j = 0; j < (size_t)next_size; j++) {
            double sum = biases[i][j];

            for (size_t k = 0; k < topology[i]; k++) {
                sum += activations[i][k] * weights[i][k][j];
            }

            if (isOutputLayer) {
                activations[i + 1][j] = 1.0 / (1.0 + exp(-sum));
            } else {
                activations[i + 1][j] = ReLU(sum);
            }
        }
    }

    return activations.back();
}

void NeuralNetwork::train(const std::vector<double>& inputs, const std::vector<double>& targets) {
    if (targets.size() != topology.back()) {
        throw std::invalid_argument("Target size mismatch with output layer!");
    }

    predict(inputs);

    for (auto& grad_layer : gradients) {
        std::fill(grad_layer.begin(), grad_layer.end(), 0.0);
    }

    int lastLayer = topology.size() - 1;

    for (int j = 0; j < topology[lastLayer]; j++) {
        double out = activations[lastLayer][j];
        double error = targets[j] - out;

        gradients[lastLayer][j] = error * (out * (1.0 - out));
    }

    for (int i = lastLayer - 1; i > 0; i--) {
        for (int j = 0; j < topology[i]; j++) {
            double error = 0;
            const auto& next_grad_layer = gradients[i+1];
            const auto& current_weight_row = weights[i][j];

            for (int k = 0; k < topology[i + 1]; k++) {
                error += next_grad_layer[k] * current_weight_row[k];
            }

            gradients[i][j] = error * ReLUDerivative(activations[i][j]);
        }
    }

    for (size_t i = 0; i < weights.size(); i++) {
        const auto& current_grad = gradients[i+1];
        const auto& current_act = activations[i];
        int next_size = (int)topology[i + 1];
        
        #pragma omp parallel for if(next_size > 512)
        for (size_t k = 0; k < (size_t)next_size; k++) {
            double grad_val = current_grad[k];
            double lr_grad = learningRate * grad_val;
            
            biases[i][k] += lr_grad;
            
            for (size_t j = 0; j < (size_t)topology[i]; j++) {
                weights[i][j][k] += lr_grad * current_act[j];
            }
        }
    }
}