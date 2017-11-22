#ifndef NEURAL_NET_HPP_
#define NEURAL_NET_HPP_

#include <armadillo>
#include <config.h>
#include <string>
#include "util.hpp"
#include "mnist.hpp"

#include <algorithm>
#include <random>
#include <vector>
#include <numeric>

// Apply sigmoid to all elements
template<typename T>
void sigmoid(T&& out) {
    out.transform([](typename T::elem_type val) {
        return (1.0 / (1.0 + exp (-1.0 * val)));
    });
}

/**
 * Simple two-layer neural network for the MNIST data set
 *
 * \code
 * auto train_net = neural_net<float>(train_data, 0.1);
 * \endcode
 */
template<typename T = float>
struct neural_net {
    using elem_t = T;
    using mat_t = arma::Mat<elem_t>;
    using vec_t = arma::Col<elem_t>;
    using subview_t = arma::subview<elem_t>;
    using mnist = mnist<elem_t>;

    const mnist input;    //< Input data
    const elem_t lambda;  //< Regularization parameter
    mat_t theta1;         //< Weights for layer 1
    mat_t theta2;         //< Weights for layer 2
    mat_t a1;             //< Input activations
    mat_t a2;             //< Output activations from layer 1
    mat_t a3;             //< Output activations from layer 2
    mat_t yy;             //< One-hot vector for labels
    mat_t d_theta1;       //< Gradient for layer 1 weights
    mat_t d_theta2;       //< Gradient for layer 2 weights
    mat_t delta3;         //< Gradient from output to labels
    mat_t delta2;         //< Gradient from layer 2 to layer 1 activations

    /**
     * Constructor
     *
     * @param input_   MNIST inut data
     * @param lambda_  lambda value for regularization. Set to 0 for no
     *                 regularization
     */
    neural_net(const mnist& input_, elem_t lambda_ = 1)
    : input(input_),
      lambda(lambda_) {
        using rowvec = arma::Row<elem_t>;
        using arma::zeros;
        using arma::ones;
        using arma::randu;

        elem_t epsilon = 0.12;
        theta1 = (randu<mat_t>(64, input.images.n_cols + 1) * 2 - 1) * epsilon;
        theta2 = (randu<mat_t>(10, theta1.n_rows + 1) * 2 - 1) * epsilon;

        // Convert the column vector of y labels to a matrix where each
        // row has a 1 in the column specified by the label.
        yy = zeros<mat_t>(input.labels.n_rows, 11);
        int row = 0;
        input.labels.for_each([&](const elem_t& element) {
            yy(row, element) = 1.0;
            yy(row, 0) = 1.0;
            row++;
        });

        // Allocate space for each layer of activations. Add an additional
        // column for the bias neuron
        a1 = ones<mat_t>(input.images.n_rows, input.images.n_cols + 1);
        a2 = ones<mat_t>(input.images.n_rows, theta1.n_rows + 1);
        a3 = ones<mat_t>(input.images.n_rows, theta2.n_rows + 1);

        // Load the images onto first activation layer, but do not overwrite
        // the bias neuron
        no_bias(a1) = input.images;

        //shuffle_a1_yy();
    }


    /**
     * Use the neural network to predict outcomes
     *
     * @return The percentage of correct predictions

     */
    elem_t predict(void) const {
        feed_forward();

        // Find neuron with maximum confidence, this is our predicted label
        arma::ucolvec predictions = index_max(no_bias(a3), 1);
        predictions = predictions + 1;

        // Display percentage of correct labels
        return elem_t(sum(predictions == input.labels)) / input.labels.n_rows;
    }

    /**
     * Cost function for the neural net
     *
     * @return the cost
     */
    elem_t cost() const {
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

    /**
     * Calculate the gradient for the weights
     */
    void gradient() {
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

    /**
     * A single Forward propogation step for the neural net
     */
    void feed_forward() const {
        // Pass input thru weights to second layer, do not disturb the bias
        // neuron
        no_bias(a2) = a1 * theta1.t();
        sigmoid(no_bias(a2));

        // Pass hidden layer to output layer
        no_bias(a3) = a2 * theta2.t();
        sigmoid(no_bias(a3));
    }

    /**
     * Train the neural network. Pass in a number of steps and a progress
     * function.
     *
     * \code
     * net.train(1000, [&](size_t i, size_t max_itr) -> void {
     *     if (i == 0 || i % 100 == 0 || i == max_itr - 1) {
     *         std::cout << "\r " << i << " j = " << train_net.cost()
     *                   << std::flush;
     *     }
     * });
     * \endcode
     *
     * @param steps    The number of steps to train
     * @param progress Function called after each step.
     */
    void train(size_t steps, std::function<void(size_t, size_t)> progress) {
        for (size_t i = 0; i < steps; i++) {
            feed_forward();
            // Back prop
            gradient();
            theta1 -= d_theta1;
            theta2 -= d_theta2;
            progress(i, steps);
        }
    }

    /**
     * Save the weights to two files
     */
    void save() const {
        std::string theta1_fn = data_dir + "/theta1.bin";
        std::string theta2_fn = data_dir + "/theta2.bin";
        theta1.save(theta1_fn);
        theta2.save(theta2_fn);
    }

    /**
     * Shuffle the rows of the input and label matrices
     */
    void shuffle_a1_yy() {
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
};

#endif /* end of include guard: NEURAL_NET_HPP_ */
