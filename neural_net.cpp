#include "neural_net.hpp"
#include "png_weight.hpp"

//using namespace arma;

using arma::randu;
using arma::ones;
using arma::rowvec;
using arma::zeros;
using arma::ucolvec;

neural_net::neural_net(const mnist& input_, elem_t lambda_)
: input(input_),
  lambda(lambda_) {

    elem_t epsilon = 0.12;
    theta1 = (randu(25, input.images.n_cols + 1) * 2 - 1) * epsilon;
    theta2 = (randu(10, theta1.n_rows + 1) * 2 - 1) * epsilon;

    // Convert the column vector of y labels to a matrix where each
    // row has a 1 in the column specified by the label.
    yy.resize(input.labels.n_rows, 11);
    int idx = 0;
    input.labels.for_each([&](const elem_t& element) {
        rowvec label = zeros<rowvec>(11);
        label[element] = 1.0;
        label[0] = 1.0;
        yy.row(idx) = label;
        idx++;
    });

    a1 = ones(input.images.n_rows, input.images.n_cols + 1);
    no_bias(a1) = input.images;
    a2 = ones(input.images.n_rows, theta1.n_rows + 1);
    a3 = ones(input.images.n_rows, theta2.n_rows + 1);
}

void neural_net::save() const {
    std::string theta1_fn = data_dir + "/theta1.bin";
    std::string theta2_fn = data_dir + "/theta2.bin";
    theta1.save(theta1_fn);
    theta2.save(theta2_fn);
}

void neural_net::feed_forward() {
    // Pass input thru weights to second layer
    no_bias(a2) = a1 * theta1.t();
    sigmoid(no_bias(a2));

    // Pass hidden layer to output layer
    no_bias(a3) = a2 * theta2.t();
    sigmoid(no_bias(a3));
}

neural_net::elem_t neural_net::cost() const {
    // Cost, without regularization
    elem_t cost = sum(sum(
        -1 * no_bias(yy) % log(no_bias(a3)) -
            (1 - no_bias(yy)) % log(1 - no_bias(a3)), 1))
        / input.images.n_rows;

    // Cost, with regularization
    if (std::abs(lambda) > 0) {
        // Square each element. This next operation makes a copy
        mat_t theta1_sq = square(no_bias(theta1));
        mat_t theta2_sq = square(no_bias(theta2));

        // Sum up all elements in each layer
        elem_t reg = (sum(sum(theta1_sq)) + sum(sum(theta2_sq)));

        // Normalize
        reg *= lambda / (2 * input.images.n_rows);

        // Add in regularization term
        cost += reg;
    }

    return cost;
}

void neural_net::gradient() {
    elem_t m = a1.n_rows;

    // Delta from labeled data to current set of predictions
    delta3 = a3 - yy;
    d_theta2 = (no_bias(delta3).t() * a2) / m;

    // Delta from output layer to hidden layer
    delta2 = no_bias(delta3) * theta2 % (a2 % (1 - a2));

    // Delta from output layer to hidden layer without bias
    d_theta1 = (no_bias(delta2).t() * a1) / m;

    // TODO: Regularization
    // if (std::abs(lambda) > 0) {
    //     // Operate on subviews so as to not disturb the weights that connect
    //     // to the bias neuron
    //     subview_t theta1_nb = no_bias(theta1);
    //     subview_t theta2_nb = no_bias(theta2);
    //     subview_t grad1_sv = no_bias(d_theta1);
    //     subview_t grad2_sv = no_bias(d_theta2);
    //
    //     grad1_sv = grad1_sv + theta1_nb * (lambda / m);
    //     grad2_sv = grad2_sv + theta2_nb * (lambda / m);
    // }
}

// Predict the label for each image
void neural_net::predict() {
    feed_forward();

    // Find neuron with maximum confidence, this is our predicted label
    ucolvec predictions = index_max(no_bias(a3), 1);
    predictions = predictions + 1;

    // Display percentage of correct labels
    elem_t percentage = elem_t(sum(predictions == input.labels)) / input.labels.n_rows;
    std::cout << "precentage correct: " << percentage << std::endl;
}

void neural_net::train(size_t max_itr) {

    for (size_t i = 0; i < max_itr; i++) {
        feed_forward();
        // Back prop
        gradient();
        theta1 -= d_theta1;
        theta2 -= d_theta2;
        // Cost
        if (i == 0 || i % 100 == 0 || i == max_itr - 1) {
            std::cout << "\r " << i << " j = " << cost() << std::flush;
        }
    }
    std::cout << std::endl;

    weight_png("theta1.png", theta1, 28, 28, 2, 2);
    weight_png("theta2.png", theta2, 5, 5, 2, 2);
}
