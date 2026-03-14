#pragma once

#include <vector>
#include <stdexcept>

double sigmoid(double x);
double sigmoidDerivative(double y);

class NeuralNetwork {
private:
    std::vector<int> topology;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<double>> activations;
    double learningRate;

    double getRandom() const;

public:   
    NeuralNetwork(std::vector<int> topology, double lr = 0.1);
    double predict(const std::vector<double>& inputs);
    void train(const std::vector<double>& inputs, double target);
};