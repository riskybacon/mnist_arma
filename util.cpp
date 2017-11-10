#include <sstream>
#include "util.hpp"

std::ifstream open_file(const std::string& filename) {
    std::ifstream file;

    // Set exceptions to be thrown on failure
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try {
        file.open(filename);
    } catch (std::system_error& e) {
        std::stringstream err;
        err << "Error opening '" << filename << "': " << e.code().message();
        throw std::runtime_error(err.str());
    }

    return file;
}

size_t reverse_int(size_t i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((size_t)ch1<<24)+((size_t)ch2<<16)+((size_t)ch3<<8)+ch4;
}
