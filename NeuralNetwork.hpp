#pragma once

#include <vector>
#include <stdexcept>

double ReLU(double x);
double ReLUDerivative(double y);

class NeuralNetwork {
private:
    std::vector<int> topology;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> gradients;
    double learningRate;

    double getRandom() const;

public:   
    NeuralNetwork(std::vector<int> topology, double lr = 0.001);
    std::vector<double> predict(const std::vector<double>& inputs);
    void train(const std::vector<double>& inputs, const std::vector<double>& targets);
    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);
};