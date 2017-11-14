#ifndef MNIST_HPP_
#define MNIST_HPP_

#include <armadillo>
#include "util.hpp"

/**
 * MNIST data storage object.
 *
 * \code
 * string train_images_fn = data_dir + "/train-images-idx3-ubyte";
 * string train_labels_fn = data_dir + "/train-labels-idx1-ubyte";
 * auto train_data = mnist<float>::create(
 *     train_images_fn, train_labels_fn);
 * \endcode
 */
template<typename T = float>
struct mnist {
    using elem_t = T;
    using mat_t = arma::Mat<elem_t>;
    using col_t = arma::Col<elem_t>;

    const size_t size;      //< Number of images in the set
    const size_t width;     //< Width of a single image
    const size_t height;    //< Height of a single image
    const size_t channels;  //< Number of channels in the image
    const mat_t images;     //< Image data
    const col_t labels;     //< Label data

    /**
     * Creates an MNIST object that contains images and labels
     *
     * @param images_fn  Filename for MNIST images
     * @param labels_fn  Filename for MNIST labels
     * @return An mnist object
     */
    static mnist create(
        const std::string& images_fn,
        const std::string& labels_fn
    ) {
        using std::tie;

        mat_t images;
        col_t labels;
        size_t width;
        size_t height;

        tie(images, width, height) = load_images(images_fn);
        labels = load_labels(labels_fn);

        return mnist(images, labels, width, height);
    }

private:

    /**
     * Constructor. Use the create() function to build an mnist object
     *
     * @param images_  A matrix of images, one image per row
     * @param labels_  A column vector of labels, each entry is 0-9
     * @param width_   The width of each image
     * @param height_  The height of each image
     */
    mnist(const mat_t& images_, const col_t& labels_, size_t width_,
      size_t height_)
    : images(images_), labels(labels_), width(width_), height(height_),
      size(labels_.n_rows), channels(1) {}

    /**
     * Load mnist labels from a file.
     *
     * @param filename  The filename of the label file
     * @return the labels in a column vector
     */
    static col_t load_labels(const std::string& filename) {
        std::ifstream file = open_file(filename);

        uint32_t magic_number = 0;
        uint32_t num_images = 0;

        file.read((char*) &magic_number, sizeof(magic_number));
        file.read((char*) &num_images, sizeof(num_images));

        magic_number = reverse_int(magic_number);
        num_images = reverse_int(num_images);

        col_t labels(num_images);

        for(size_t img = 0; img < num_images; ++img) {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            labels(img) = elem_t(temp);
        }

        return labels;
    }

    /**
     * Load mnist images from a file. A tuple is returned that contains a
     * matrix of the images, one image per row, the height of a single image,
     * and the width of a single image.
     *
     * @param filename  The filename of the image file
     * @return a tuple: image matrix, width, height
     */
    static std::tuple<mat_t, size_t, size_t> load_images(
      const std::string& filename) {
        std::ifstream file = open_file(filename);

        uint32_t magic_number = 0;
        uint32_t num_images = 0;
        uint32_t num_rows = 0;
        uint32_t num_cols = 0;
        size_t num_pixels = 0;

        file.read((char*) &magic_number, sizeof(magic_number));
        file.read((char*) &num_images, sizeof(num_images));
        file.read((char*) &num_rows, sizeof(num_rows));
        file.read((char*) &num_cols, sizeof(num_cols));

        magic_number = reverse_int(magic_number);
        num_images = reverse_int(num_images);
        num_rows = reverse_int(num_rows);
        num_cols = reverse_int(num_cols);
        num_pixels = num_rows * num_cols;

        mat_t data(num_images, num_pixels);

        for (size_t img = 0; img < num_images; ++img) {
            for (size_t idx = 0; idx < num_pixels; ++idx) {
                unsigned char temp = 0;
                file.read((char*) &temp, sizeof(temp));
                data(img, idx) = elem_t(temp) / elem_t(255);
            }
        }

        return std::make_tuple(data, num_cols, num_rows);
      }

};

#endif /* end of include guard: MNIST_HPP_ */
