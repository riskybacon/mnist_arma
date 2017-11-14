#ifndef UTIL_HPP_
#define UTIL_HPP_

#include <fstream>
#include <string>
#include <armadillo>

std::ifstream open_file(const std::string& filename);
size_t reverse_int(size_t i);

// Return subview without bias column
template<typename T = arma::mat>
arma::subview<typename T::elem_type> no_bias(T& val){
    return val.tail_cols(val.n_cols - 1);
}

#endif /* end of include guard: UTIL_HPP_ */
