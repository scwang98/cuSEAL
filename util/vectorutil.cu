//
// Created by scwang on 2023/11/12.
//

#include "vectorutil.h"
#include "../extern/cnpy/cnpy.h"
#include <cmath>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/Clustering.h>

namespace util {

    double calculateMagnitude(const std::vector<float> &vector) {
        double sum = 0.0;
        for (double value: vector) {
            sum += value * value;
        }
        return std::sqrt(sum);
    }

    void normalizeVector(std::vector<float> &vector) {
        float magnitude = calculateMagnitude(vector);
        if (magnitude > 0.0) {
            for (float &value: vector) {
                value /= magnitude;
            }
        }
    }

    float calculateMagnitude(const float *array, size_t size) {
        float sum = 0.0;
        for (int i = 0; i < size; i++) {
            auto value = array[i];
            sum += value * value;
        }
        return std::sqrt(sum);
    }

    void normalizeArray(float *array, size_t size) {
        float magnitude = calculateMagnitude(array, size);
        if (magnitude > 0.0) {
            for (int i = 0; i < size; i++) {
                array[i] /= magnitude;
            }
        }
    }

    std::vector<std::vector<float>> read_npy_data(const std::string &npy_name) {
        cnpy::NpyArray arr = cnpy::npy_load(npy_name);
        std::vector<size_t> shape = arr.shape;
        size_t numRows = shape[0];
        size_t numCols = shape[1];
        auto *data = arr.data<float>();
        std::vector<std::vector<float>> matrix;
        matrix.reserve(numRows);
        for (size_t i = 0; i < numRows; ++i) {
            std::vector<float> row;
            row.reserve(numCols);
            for (size_t j = 0; j < numCols; ++j) {
                auto value = data[i * numCols + j];
                row.push_back(value);
            }
            normalizeArray(row.data(), row.size());
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

    void kmeans_clustering(
            size_t dimension,
            const float *training_set,
            size_t training_set_size,
            std::vector<std::vector<float>> &cluster,
            std::vector<std::vector<int64_t>> &indexes,
            std::vector<float> &centroids,
            size_t centroids_size) {
        faiss::Clustering clus(dimension, centroids_size);
        faiss::IndexFlatIP quantizer(dimension);
        clus.train(training_set_size, training_set, quantizer);
        memcpy(centroids.data(), clus.centroids.data(), sizeof(float) * dimension * centroids_size);

        faiss::IndexIVFFlat index(&quantizer, dimension, centroids_size);
        index.add(training_set_size, training_set);

        auto inv_lists = dynamic_cast<faiss::ArrayInvertedLists *>(index.invlists);
        for (int i = 0; i < centroids_size; i++) {
            auto &inv_data = inv_lists->codes[i];
            auto &clu_data = cluster[i];
            auto inv_len = sizeof(unsigned char) * inv_data.size();
            clu_data.resize(inv_len / sizeof(float));
            memcpy(clu_data.data(), inv_data.data(), inv_len);

            indexes[i] = inv_lists->ids[i];
        }
    }

    // matrix, capacity, size
    std::tuple<float *, size_t, size_t>
    format_matrix(std::vector<float> &vec, size_t batch_size, size_t single_size, float scale) {
        auto origin_size = vec.size() / single_size;
        auto slots = single_size * batch_size;
        size_t size = origin_size / slots * single_size;
        auto end_row = size;
        if (origin_size % slots != 0) {
            size += single_size;
        }
        auto data = vec.data();
        auto matrix = new float[size * slots];
        for (size_t offset = 0; offset < end_row; offset += single_size) {
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
        if (size != end_row) {
            auto matrix_start = matrix + (end_row * slots);
            std::fill_n(matrix_start, single_size * slots, 0);
            auto data_start = data + (end_row * slots);

            for (size_t i = 0; i < origin_size - end_row * (slots / single_size); i++) {
                auto row_ptr = matrix_start + i;
                auto data_ptr = data_start + i * single_size;
                for (size_t j = 0; j < single_size; j++) {
                    *(row_ptr + j * slots) = *(data_ptr + j) * scale;
                }
            }
        }

        return std::tuple{matrix, size, origin_size};
    }


    std::vector<std::tuple<float *, size_t, size_t>>
    read_cluster_npy_data(const std::string &npy_name, size_t slots, float scale, size_t centroids_size, std::vector<float> &centroids, std::vector<std::vector<int64_t>> &indexes) {
        cnpy::NpyArray arr = cnpy::npy_load(npy_name);
        std::vector<size_t> shape = arr.shape;
        size_t single_size = shape[1];
        size_t vector_count = shape[0];
        auto batch_size = slots / single_size;

        auto data = arr.data<float>();

        for (int i = 0; i < vector_count; i++) {
            normalizeArray(data + i * single_size, single_size);
        }

        std::vector<std::vector<float>> cluster(centroids_size);
        indexes.resize(centroids_size);
        centroids.resize(centroids_size * single_size);
        kmeans_clustering(single_size, data, shape[0], cluster, indexes, centroids, centroids_size);

        std::vector<std::tuple<float *, size_t, size_t>> formatted_cluster;
        formatted_cluster.reserve(cluster.size());
        for (uint i = 0; i < cluster.size(); i++) {
            formatted_cluster.push_back(format_matrix(cluster[i], batch_size, single_size, scale));
        }
//        for (auto& vec : cluster) {
//            formatted_cluster.push_back(format_matrix(vec, batch_size, single_size, scale));
//        }
        cluster.clear();

        return formatted_cluster;
    }

    float *read_formatted_npy_data(const std::string &npy_name, size_t slots, float scale, size_t &size) {
        cnpy::NpyArray arr = cnpy::npy_load(npy_name);
        std::vector<size_t> shape = arr.shape;
        size_t single_size = 512;
        auto batch_size = slots / single_size;
        size_t numRows = shape[0] / batch_size; // TODO: 未读取完整 @wangshuchao
        size_t numCols = shape[1] * batch_size;
        size = numRows;

        auto data = arr.data<float>();
        auto matrix = new float[numRows * numCols];

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

    FILE *read_npy_header(const std::string &npy_name, size_t slots, double scale, size_t &size) {
        FILE *fp = fopen(npy_name.c_str(), "rb");
        std::vector<size_t> shape;
        size_t word_size;
        bool fortran_order;
        cnpy::parse_npy_header(fp, word_size, shape, fortran_order);

        cnpy::NpyArray arr(shape, word_size, fortran_order);
        size_t nread = fread(arr.data<char>(), 1, arr.num_bytes(), fp);
        if (nread != arr.num_bytes())
            throw std::runtime_error("load_the_npy_file: failed fread");

        return fp;
    }

    NPYReader::NPYReader(const std::string &npy_name, size_t slots) {
        fp_ = fopen(npy_name.c_str(), "rb");
        std::vector<size_t> shape;
        bool fortran_order;
        cnpy::parse_npy_header(fp_, word_size_, shape, fortran_order);
//        size_t single_size = 512;
        auto batch_size = slots / 512;
        row_ = shape[0] / batch_size;
        col_ = slots;

        temp_ = new float[slots * 512];
    }

    sigma::util::HostGroup<float> *NPYReader::read_data(float scale) {
        size_t read_size = col_ * 4 * 512;
        size_t n_read = fread(temp_, 1, read_size, fp_);
        if (read_size != n_read) {
            return nullptr;
        }
        float *data = nullptr;
        cudaMallocHost((void **) &data, col_ * 512 * sizeof(float));

        auto data_start = data;
        auto temp_start = temp_;
        for (size_t i = 0; i < 512; i++) {
            auto line_ptr = data_start + (i * col_);
            auto data_ptr = temp_start + i;
            for (size_t j = 0; j < col_; j++) {
                *(line_ptr + j) = *(data_ptr + j * 512) * scale;
            }
        }

        return new sigma::util::HostGroup<float>(data, 512, col_);
    }

    NPYReader::~NPYReader() {
        delete[] temp_;
    }

} // util