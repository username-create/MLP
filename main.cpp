#include "NeuralNetwork.hpp"

#include <iostream>
#include <vector>
#include <ctime>
#include <iomanip>

#include <iostream>
#include <vector>
#include "NeuralNetwork.hpp"

int main() {
    srand(time(NULL));

    try {
        // ============================================================
        // Neural Network Topology Configuration: { Input, [Hidden...], Output }
        // ------------------------------------------------------------
        // 1. Input Layer (1st element): Must match your inputs[i].size().
        // 2. Hidden Layers (Middle elements): Scalable in both Depth and Width.
        //    - Add more elements to increase "Depth" (Number of layers).
        //    - Increase the values to increase "Width" (Neurons per layer).
        // 3. Output Layer (Last element): Must match the number of targets.
        // ============================================================
        NeuralNetwork nn({2, 2, 1});

        // Training Data
        std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

        std::vector<double> targets = {0, 1, 1, 0};

        for (int epoch = 0; epoch < 100000; epoch++) {
            int r = rand() % 4; 
            nn.train(inputs[r], targets[r]);
        }

        // Test and Output Results
        std::cout << std::fixed << std::setprecision(4);
        for (size_t i = 0; i < inputs.size(); i++) {
            double output = nn.predict(inputs[i]);   
            int logic_result = (output > 0.5) ? 1 : 0;

            std::cout << "Input: (" << (int)inputs[i][0] << ", " << (int)inputs[i][1] << ") " 
                      << "| Target: " << (int)targets[i] << " | " 
                      << "Predict: " << output << " -> "
                      << "Result: [" << logic_result << "]" << "\n";
        }
    } 
    catch (const std::invalid_argument& e) {
        // Handle architectural or input mismatches
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}