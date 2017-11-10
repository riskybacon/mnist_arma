#ifndef MNIST_HPP_
#define MNIST_HPP_

#include <armadillo>

struct mnist {
    using vec_t = arma::colvec;
    using mat_t = arma::mat;
    using elem_t = mat_t::elem_type;

    const size_t size;
    const size_t width;
    const size_t height;
    const size_t channels;
    const mat_t images;
    const vec_t labels;

    static mnist create(const std::string& images, const std::string& labels);

private:
    mnist(const mat_t& images_, const vec_t& labels_, size_t width_,
      size_t height_)
    : images(images_), labels(labels_), width(width_), height(height_),
      size(labels_.n_rows), channels(1) {}

    static vec_t load_labels(const std::string& filename);
    static std::tuple<mat_t, size_t, size_t> load_images(
      const std::string& filename);

};

#endif /* end of include guard: MNIST_HPP_ */
