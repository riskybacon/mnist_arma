#include <algorithm>
#include <random>
#include <vector>
#include <numeric>

#include "neural_net.hpp"
#include "png_weight.hpp"

using arma::randu;
using arma::ones;
using arma::rowvec;
using arma::zeros;
using arma::ucolvec;

neural_net::neural_net(const mnist& input_, elem_t lambda_)
: input(input_),
  lambda(lambda_) {

    elem_t epsilon = 0.12;
    theta1 = (randu(64, input.images.n_cols + 1) * 2 - 1) * epsilon;
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

    // Allocate space for each layer of activations. Add an additional column
    // for the bias neuron
    a1 = ones(input.images.n_rows, input.images.n_cols + 1);
    a2 = ones(input.images.n_rows, theta1.n_rows + 1);
    a3 = ones(input.images.n_rows, theta2.n_rows + 1);

    // Load the images onto first activation layer, but do not overwrite the
    // bias neuron
    no_bias(a1) = input.images;

    //shuffle_a1_yy();
}

void neural_net::shuffle_a1_yy() {
    using std::iota;
    using std::vector;
    using std::shuffle;
    using std::begin;
    using std::end;

    vector<size_t> indices(a1.n_rows);
    iota(begin(indices), end(indices), 0);

    auto rng = std::default_random_engine {};
    shuffle(begin(indices), end(indices), rng);

    mat_t a1_shfl(a1.n_rows, a1.n_cols);
    mat_t yy_shfl(yy.n_rows, yy.n_cols);

    for (size_t dst_idx = 0; dst_idx < indices.size(); ++dst_idx) {
        size_t src_idx = indices[dst_idx];
        a1_shfl.row(dst_idx) = a1.row(src_idx);
        yy_shfl.row(dst_idx) = yy.row(src_idx);
    }

    a1 = a1_shfl;
    yy = yy_shfl;
}

void neural_net::save() const {
    std::string theta1_fn = data_dir + "/theta1.bin";
    std::string theta2_fn = data_dir + "/theta2.bin";
    theta1.save(theta1_fn);
    theta2.save(theta2_fn);
}

void neural_net::feed_forward() const {
    // Pass input thru weights to second layer, do not disturb the bias neuron
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

    // Regularization
    if (std::abs(lambda) > 0) {
        no_bias(d_theta1) += (lambda / m) * no_bias(theta1);
        no_bias(d_theta2) += (lambda / m) * no_bias(theta2);
    }
}

// Predict the label for each image
neural_net::elem_t neural_net::predict() const {
    feed_forward();

    // Find neuron with maximum confidence, this is our predicted label
    ucolvec predictions = index_max(no_bias(a3), 1);
    predictions = predictions + 1;

    // Display percentage of correct labels
    return elem_t(sum(predictions == input.labels)) / input.labels.n_rows;
}

void neural_net::train(size_t max_itr,
  std::function<void(size_t, size_t)> progress) {
    for (size_t i = 0; i < max_itr; i++) {
        feed_forward();
        // Back prop
        gradient();
        theta1 -= d_theta1;
        theta2 -= d_theta2;
        progress(i, max_itr);
    }
}
