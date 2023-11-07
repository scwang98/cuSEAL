
#include "extern/cnpy/cnpy.h"

#include <cmath>
#include <string>
#include <vector>

void read_npy_label_array(const std::string& npy_name, int* result, int read_num) {
    cnpy::NpyArray arr = cnpy::npy_load(npy_name);
    int *loaded_data = arr.data<int>();
    for (int i = 0; i < read_num; ++i) {
        *(result + i) = loaded_data[i];
    }
}


double calculateMagnitude(const std::vector<double> &vector) {
    double sum = 0.0;
    for (double value: vector) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

void normalizeVector(std::vector<double> &vector) {
    double magnitude = calculateMagnitude(vector);
    if (magnitude > 0.0) {
        for (double &value: vector) {
            value /= magnitude;
        }
    }
}

std::vector<std::vector<double>> read_npy_data(const std::string &npy_name) {
    cnpy::NpyArray arr = cnpy::npy_load(npy_name);
    std::vector<size_t> shape = arr.shape;
    size_t numRows = shape[0];
    size_t numCols = shape[1];
    auto *data = arr.data<float>();
    std::vector<std::vector<double>> matrix;
    matrix.reserve(numRows);
    for (size_t i = 0; i < numRows; ++i) {
        std::vector<double> row;
        row.reserve(numCols);
        for (size_t j = 0; j < numCols; ++j) {
            auto value = (double) data[i * numCols + j];
            row.push_back(value);
        }
        normalizeVector(row);
        matrix.push_back(row);
    }
    return matrix;
}
