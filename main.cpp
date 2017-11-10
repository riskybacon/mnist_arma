#include <iostream>
#include <armadillo>
#include <config.h>
#include <string>
#include <pngwriter.h>
#include <assert.h>
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

    auto train_net = neural_net(train_data);
    train_net.train(1000);

    auto test_net = neural_net(test_data);
    test_net.theta1 = train_net.theta1;
    test_net.theta2 = train_net.theta2;
    train_net.predict();
}

int main(int, char**) {
    try {
        run();
    } catch (std::runtime_error err) {
        std::cerr << err.what() << std::endl;
    }
}
