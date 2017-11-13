#include <iostream>
#include <armadillo>
#include <config.h>
#include <string>
#include "png_weight.hpp"
#include "util.hpp"
#include "neural_net.hpp"
#include "mnist.hpp"

using std::string;

void run() {
    string train_images_fn = data_dir + "/train-images-idx3-ubyte";
    string train_labels_fn = data_dir + "/train-labels-idx1-ubyte";
    string test_images_fn = data_dir + "/t10k-images-idx3-ubyte";
    string test_labels_fn = data_dir + "/t10k-labels-idx1-ubyte";

    auto train_data = mnist::create(train_images_fn, train_labels_fn);
    auto test_data = mnist::create(test_images_fn, test_labels_fn);
    auto train_net = neural_net(train_data, 0.1);

    // Train the neural net. Pass in a lambda to display progress
    train_net.train(10000, [&](size_t i, size_t max_itr) -> void {
        // Display progress
        if (i == 0 || i % 100 == 0 || i == max_itr - 1) {
            std::cout << "\r " << i << " j = " << train_net.cost()
                      << std::flush;
        }
    });

    std::cout << std::endl;

    // Create a 2nd neural network using the learned weights from training data
    auto test_net = neural_net(test_data);
    test_net.theta1 = train_net.theta1;
    test_net.theta2 = train_net.theta2;

    auto percentage = train_net.predict();

    std::cout << "precentage correct: " << percentage << std::endl;

    // Write out the weight to a png file
    weight_png("theta1.png", no_bias(train_net.theta1), 28, 28, 2, 2);
    weight_png("theta2.png", no_bias(train_net.theta2), 8, 8, 2, 2);
}

int main(int, char**) {
    try {
        run();
    } catch (std::runtime_error err) {
        std::cerr << err.what() << std::endl;
    }
}
