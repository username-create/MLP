#include "NeuralNetwork.hpp"

#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>

bool loadMNIST(const std::string& filename, 
               std::vector<std::vector<double>>& inputs, 
               std::vector<std::vector<double>>& targets,
               int limit = 5000) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    std::getline(file, line); 

    int count = 0;
    while (std::getline(file, line) && count < limit) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string value;

        if (!std::getline(ss, value, ',')) continue;
        
        try {
            int label = std::stoi(value);
            
            std::vector<double> target(10, 0.0);
            if (label >= 0 && label <= 9) {
                target[label] = 1.0;
            } else {
                continue;
            }
            targets.push_back(target);

            std::vector<double> input;
            while (std::getline(ss, value, ',')) {
                input.push_back(std::stod(value) / 255.0);
            }
            inputs.push_back(input);
            count++;
        } catch (const std::exception& e) {
            std::cerr << "Skip invalid line at count " << count << " Error: " << e.what() << "\n";
            continue; 
        }
    }
    return true;
}

void printDigit(const std::vector<double>& input) {
    for (int i = 0; i < 784; i++) {
        if (input[i] > 0.7) {
            std::cout << "@@";
        } else if (input[i] > 0.3) {
            std::cout << "..";
        } else {
            std::cout << "  ";
        }
        
        if ((i + 1) % 28 == 0) std::cout << "\n";
    }
}

int main() {
    try {
        #ifdef _WIN32
            system("cls");
        #endif

        // ============================================================
        // Neural Network Topology Configuration: { Input, [Hidden...], Output }
        // ------------------------------------------------------------
        // 1. Input Layer (1st element): Must match your inputs[i].size().
        // 2. Hidden Layers (Middle elements): Scalable in both Depth and Width.
        //    - Add more elements to increase "Depth" (Number of layers).
        //    - Increase the values to increase "Width" (Neurons per layer).
        // 3. Output Layer (Last element): Must match the number of targets.
        // ============================================================
        NeuralNetwork nn({784, 128, 64, 10}, 0.001);
        std::random_device rd;
        std::mt19937 g(rd());

        std::string modelFile = "mnist_model.txt";
        std::ifstream checkFile(modelFile);

        if (checkFile.good()) {
            checkFile.close();
            std::cout << "--- Found existing model. Loading and continuing training... ---" << "\n";
            nn.loadModel(modelFile);
        } else {
            std::cout << "--- No existing model found. Starting from scratch... ---" << "\n";
        }

        std::vector<std::vector<double>> trainInputs, trainTargets;
        std::cout << "Loading Training Data (mnist_train.csv)..." << "\n";
        if (!loadMNIST("mnist_train.csv", trainInputs, trainTargets, 60000)) {
            std::cerr << "Error: Could not find mnist_train.csv" << "\n";
            return 1;
        }

        std::vector<std::vector<double>> testInputs, testTargets;
        std::cout << "Loading Test Data (mnist_test.csv)..." << "\n";
        if (!loadMNIST("mnist_test.csv", testInputs, testTargets, 10000)) {
            std::cerr << "Error: Could not find mnist_test.csv" << "\n";
            return 1;
        }

        std::vector<int> indices(trainInputs.size());
        std::iota(indices.begin(), indices.end(), 0);

        int total_epochs = 5;
        std::cout << "Starting Training (" << total_epochs << " Epochs)..." << "\n";

        auto start = std::chrono::high_resolution_clock::now();

        for (int epoch = 1; epoch <= total_epochs; epoch++) {
            
            std::shuffle(indices.begin(), indices.end(), g);

            for (size_t i = 0; i < indices.size(); i++) {
                int idx = indices[i];
                nn.train(trainInputs[idx], trainTargets[idx]);

                if (i % 500 == 0 || i == indices.size() - 1) {
                    double percentage = (double)(i + 1) / indices.size() * 100.0;
                    std::cout << "\rEpoch " << epoch << " | Progress: " 
                              << std::fixed << std::setprecision(1) << percentage << "% " 
                              << std::flush;
                }
            }

            nn.saveModel(modelFile);
            std::cout << " -> Done!" << "\n";
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "\nTotal Training Time: " << diff.count() << " seconds" << "\n";

        std::vector<double> testInput = trainInputs[0];
        std::vector<double> prediction = nn.predict(testInput);
        
        std::cout << "\n--- Final Evaluation on Unseen Data ---" << "\n";
        int testMatchCount = 0;
        std::vector<int> wrongIndices;
        
        for (size_t i = 0; i < testInputs.size(); i++) {
            std::vector<double> prediction = nn.predict(testInputs[i]);
            
            int ai_guess = std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end()));
            int real_answer = std::distance(testTargets[i].begin(), std::max_element(testTargets[i].begin(), testTargets[i].end()));

            if (ai_guess == real_answer) {
                testMatchCount++;
            } else {
                wrongIndices.push_back(i);
            }
        }

        double finalAccuracy = (testMatchCount * 100.0) / testInputs.size();
        std::cout << "Test Set Accuracy: " << std::fixed << std::setprecision(2) << finalAccuracy << "%" << "\n";
        std::cout << "Total Errors: " << wrongIndices.size() << " / 10000" << "\n";

        if (!wrongIndices.empty()) {
            std::cout << "\n--- Visualizing 5 Failed Cases ---" << "\n";
            
            std::shuffle(wrongIndices.begin(), wrongIndices.end(), g);

            int numToPrint = std::min(5, (int)wrongIndices.size());
            for (int i = 0; i < numToPrint; i++) {
                int idx = wrongIndices[i];
                std::cout << "\n[Error Case #" << i + 1 << "] Index: " << idx << "\n";
                
                printDigit(testInputs[idx]);

                std::vector<double> prediction = nn.predict(testInputs[idx]);
                int ai_guess = std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end()));
                int real_answer = std::distance(testTargets[idx].begin(), std::max_element(testTargets[idx].begin(), testTargets[idx].end()));

                std::cout << "Real: [" << real_answer << "] | AI guessed: [" << ai_guess << "]" << "\n";
                std::cout << "------------------------------------" << "\n";
            }
        }
    }
    catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}