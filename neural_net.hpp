#ifndef NEURAL_NET_HPP_
#define NEURAL_NET_HPP_

#include <iostream>
#include <armadillo>
#include <config.h>
#include <string>
#include <pngwriter.h>
#include <assert.h>
#include "util.hpp"
#include "mnist.hpp"

struct neural_net {
    using mat_t = arma::mat;
    using vec_t = arma::colvec;
    using elem_t = mat_t::elem_type;
    using subview_t = arma::subview<elem_t>;

    const mnist input;
    mat_t theta1;
    mat_t theta2, a1, a2, a3, yy, d_theta1, d_theta2;
    const elem_t lambda;
    mat_t delta2;
    mat_t delta3;

    neural_net(const mnist& input_, elem_t lambda_ = 1);
    elem_t predict(void) const;
    elem_t cost() const;
    void gradient();
    void feed_forward() const;
    void train(size_t steps, std::function<void(size_t, size_t)> progress);
    void save() const;
    void shuffle_a1_yy();
};

// Apply sigmoid to all elements
template<typename T>
void sigmoid(T&& out) {
    out.transform([](typename T::elem_type val) {
        return (1.0 / (1.0 + exp (-1.0 * val)));
    });
}

// Return subview without bias column
template<typename T = arma::mat>
arma::subview<typename T::elem_type> no_bias(T& val){
    return val.tail_cols(val.n_cols - 1);
}

#endif /* end of include guard: NEURAL_NET_HPP_ */