#include "mnist.hpp"
#include "util.hpp"

mnist mnist::create(const std::string& images_fn, const std::string& labels_fn)
{
    using std::tie;

    mnist::mat_t images;
    mnist::vec_t labels;
    size_t width;
    size_t height;

    tie(images, width, height) = load_images(images_fn);
    labels = load_labels(labels_fn);

    return mnist(images, labels, width, height);
}

mnist::vec_t mnist::load_labels(const std::string& filename) {
    using elem_t = mnist::elem_t;

    std::ifstream file = open_file(filename);

    uint32_t magic_number = 0;
    uint32_t num_images = 0;

    file.read((char*) &magic_number, sizeof(magic_number));
    file.read((char*) &num_images, sizeof(num_images));

    magic_number = reverse_int(magic_number);
    num_images = reverse_int(num_images);

    mnist::vec_t labels(num_images);

    for(size_t img = 0; img < num_images; ++img) {
        unsigned char temp = 0;
        file.read((char*) &temp, sizeof(temp));
        labels(img) = elem_t(temp);
    }

    return labels;
}

std::tuple<mnist::mat_t, size_t, size_t> mnist::load_images(
  const std::string& filename) {
    using mat_t = arma::mat;
    using elem_t = mat_t::elem_type;

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
