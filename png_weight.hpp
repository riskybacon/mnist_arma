#ifndef PNG_WEIGHT_HPP_
#define PNG_WEIGHT_HPP_

#include <pngwriter.h>
#include <assert.h>

/**
 * Write a PNG that displays the weights. The input parameters img_width and
 * img_height define the individual squares for each set of weights.
 *
 * weight.n_cols - 1 must equal img_width * img_height
 *
 * @param filename    The output filename
 * @param weight      A matrix of the weights, with a bias column
 * @param img_width   The input image width for each row
 * @param img_height  The input image height for each row
 * @param padding_x   The amount of padding between each weight square, x dim
 * @param padding_y   The amount of padding between each weight square, y dim
 */
template<typename T = arma::mat>
void weight_png(const std::string& filename, const T& weight,
  size_t img_width, size_t img_height, size_t padding_x, size_t padding_y) {
    using mat_t = T;
    using elem_type = typename T::elem_type;

    const size_t num_outputs = weight.n_rows;
    const size_t num_inputs = weight.n_cols;

    assert(num_inputs == img_width * img_height);

    // The number of images to display in X and Y dimensions
    const size_t num_images_x = std::ceil(std::sqrt(num_outputs));
    const size_t num_images_y = std::ceil(float(num_outputs) / float(num_images_x));

    const size_t width = (num_images_x * img_width) + (num_images_x + 1) * padding_x;
    const size_t height = (num_images_y * img_height) + (num_images_y + 1) * padding_y;

    pngwriter png(int(width), int(height), int(0), filename.c_str());

    const elem_type min = weight.min();
    const elem_type max = weight.max();
    const elem_type bias = -1.0 * min;
    const elem_type scale = 1.0 / (max - min);

    for (size_t img = 0; img < weight.n_rows; ++img) {
        // Get the row,col for this weight
        const size_t row = std::floor(img / float(num_images_x));
        const size_t col = img % num_images_x;
        // Get the starting x,y coords for this weight
        const size_t x = padding_x + (col * img_width) + col * padding_x;
        const size_t y = padding_y + (row * img_height) + row * padding_y;
        // Draw the weight
        draw_weight(png, weight.row(img), x, y, img_width, img_height, scale, bias);
    }

    //png.scale_k(5);
    png.close();
}

/**
 * Draw a set of weights for a single neuron at position x,y in a png
 */
template<typename T = arma::rowvec>
void draw_weight(pngwriter& png, const T& row, size_t start_x,
  size_t start_y, size_t img_width, size_t img_height,
  typename T::elem_type scale, typename T::elem_type bias) {
    using elem_t = typename T::elem_type;

    for (size_t x = 0; x < img_width; ++x) {
        for (size_t y = 0; y < img_height; ++y) {
            size_t pos_x = start_x + x;
            size_t pos_y = png.getheight() - (start_y + y);

            elem_t val = (row(y * img_width + x) + bias) * scale;

            png.plot(pos_x, pos_y, val, val, val);
        }
    }
}

#endif /* end of include guard: PNG_WEIGHT_HPP_ */
