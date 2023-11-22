//
// Created by scwang on 2023/11/12.
//

#include "vectorutil.h"
#include "../extern/cnpy/cnpy.h"

#include <cmath>

namespace util {

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

    std::vector<std::vector<double>> batch_read_npy_data(const std::string &npy_name, size_t batch_size, double scale) {
        cnpy::NpyArray arr = cnpy::npy_load(npy_name);
        std::vector<size_t> shape = arr.shape;
        size_t numRows = shape[0] / batch_size; // TODO: 未读取完整 @wangshuchao
        size_t numCols = shape[1] * batch_size;
        auto *data = arr.data<float>();
        std::vector<std::vector<double>> matrix;
        matrix.reserve(numRows);
        for (size_t i = 0; i < numRows; ++i) {
            std::vector<double> row;
            row.reserve(numCols);
            for (size_t j = 0; j < numCols; ++j) {
                auto value = (double) data[i * numCols + j] * scale;
                row.push_back(value);
            }
//            normalizeVector(row);
            matrix.push_back(row);
        }
        return matrix;
    }

    double *read_formatted_npy_data(const std::string &npy_name, size_t slots, double scale, size_t &size) {
        cnpy::NpyArray arr = cnpy::npy_load(npy_name);
        std::vector<size_t> shape = arr.shape;
        size_t single_size = 512;
        auto batch_size = slots / single_size;
        size_t numRows = shape[0] / batch_size; // TODO: 未读取完整 @wangshuchao
        size_t numCols = shape[1] * batch_size;
        size = numRows;

        auto data = arr.data<float>();
        auto matrix = new double[numRows * numCols];

//        01  02  11  12             01  11  21  31
//        21  22  31  32    ---->    02  12  22  32
//        41  42  51  52    ---->    41  51  61  71
//        61  62  71  72             42  52  62  72
        for (size_t offset = 0; offset < numRows - single_size; offset += single_size) {
            auto matrix_start = matrix + (offset * slots);
            auto data_start = data + (offset * slots);
            for (size_t i = 0; i < single_size; i++) {
                auto line_ptr = matrix_start + (i * slots);
                auto data_ptr = data_start + i;
                for (size_t j = 0; j < slots; j++) {
                    *(line_ptr + j) = *(data_ptr + j * single_size) * scale;
                }
            }
        }

        return matrix;
    }
} // util