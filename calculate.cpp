//
// Created by scwang on 2024/3/6.
//

#include "calculate.h"
#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <unordered_set>
#include <sigma.h>
#include <iomanip>

#include "extern/jsoncpp/json/json.h"
#include "util/configmanager.h"
#include "util/vectorutil.h"
#include "util/keyutil.h"
#include "util/safequeue.h"

using namespace std;

#define DIMENSION 512
#define THREAD_SIZE 8
#define PROBE_SIZE 10

std::string gallery_data_path(const std::string &directory, size_t index) {
    std::ostringstream oss;
    oss << std::setw(5) << std::setfill('0') << index;
    return directory + "/gallery_data/gallery_" + oss.str() + "_results.dat";
}

std::string ip_results_path(const std::string &directory, size_t index) {
    return directory + "/probe_" + std::to_string(index) + "_results.dat";
}

class Task {

    std::vector<float> probe_data;

public:

    size_t finished_part;

    Task() = default;

    Task(const std::vector<float>& data) : probe_data(data) {
        finished_part = 0;
    }
};

class TaskManager {
    int gpu_count;
    unordered_set<Task *> set;

public:
    vector<util::safe_queue<Task *, 20>> queues;

    TaskManager(int gpu_count) : gpu_count(gpu_count) {
        queues.resize(4);
    }

    void start_task(const std::vector<float>& data) {
        auto task = new Task(data);
        set.insert(task);
        for (uint i = 0; i < gpu_count; i++) {
            queues[i].push(task);
        }
    }

    void task_finished(Task *task) {
        task->finished_part++;
        if (task->finished_part >= gpu_count) {
            // TODO: 数据存储

            set.erase(task);
            delete task;
        }
    }
};

std::vector<sigma::Ciphertext> gallery_data;
vector<vector<vector<sigma::Ciphertext>>> gallery_data_cluster;
std::vector<std::vector<int64_t>> indexes;
std::vector<std::vector<float>> probe_data;

vector<util::safe_queue<Task *, 20>> queues;

size_t probe_index = 0;
std::mutex probe_index_mutex;

void calculate_thread(sigma::SIGMAContext &context, const sigma::Ciphertext &c1, double scale, const std::string &result_directory) {
    sigma::CKKSEncoder encoder(context);
    sigma::Evaluator evaluator(context);

    sigma::Ciphertext c1_sum;
    sigma::Ciphertext c1_row;
    sigma::Ciphertext result;
    sigma::Ciphertext row;

    std::vector<sigma::Plaintext> encoded_probes(DIMENSION);

    while (true) {
        size_t index = 0;
        {
            std::lock_guard<std::mutex> lock(probe_index_mutex);
            if (probe_index >= PROBE_SIZE) {
                break;
            }
            index = probe_index++;
        }

        const auto &probe = probe_data[index];
        // 0.022
        encoder.cu_encode(probe[0], scale, encoded_probes[0]);

        // 0.008
        evaluator.cu_multiply_plain(c1, encoded_probes[0], c1_sum);
        for (int i = 1; i < DIMENSION; ++i) {

            // 0.012
            encoder.cu_encode(probe[i], scale, encoded_probes[i]);

            // 0.006
            evaluator.cu_multiply_plain(c1, encoded_probes[i], c1_row);
            // 0.006
            evaluator.cu_add_inplace(c1_sum, c1_row);
        }
        // 0.036
        c1_sum.retrieve_to_host();

        std::ofstream ofs(ip_results_path(result_directory, index), std::ios::binary);
        // 0.07
        c1_sum.save(ofs);

        size_t calculate_size = gallery_data.size() / 512 * 512;
        for (size_t offset = 0; offset < calculate_size; offset += DIMENSION) {
            // 0.009
            evaluator.cu_multiply_plain(gallery_data[offset], encoded_probes[0], result);
            for (size_t i = 1; i < DIMENSION; i++) {
                // 0.007
                evaluator.cu_multiply_plain(gallery_data[offset + i], encoded_probes[i], row);

                // 0.007
                evaluator.cu_add_inplace(result, row);

            }

            result.retrieve_to_host();
            // 0.065
            result.save(ofs);
        }
        ofs.close();
    }
}

void task_for_gpu(int gpu_index, sigma::SIGMAContext &context, const sigma::Ciphertext &origin_c1, double scale) {
    cudaSetDevice(gpu_index);
    auto &gallery_data = gallery_data_cluster[gpu_index];
    for (auto &cluster_data : gallery_data) {
        for (auto &ciphertext : cluster_data) {
            ciphertext.copy_to_device();
        }
    }

    sigma::Ciphertext c1 = origin_c1;
    c1.copy_to_device();

    std::thread *threads[THREAD_SIZE];
    for (auto &ptr: threads) {
        ptr = new std::thread(calculate, std::ref(context), std::ref(c1), std::ref(scale));
    }

    for (auto &ptr: threads) {
        if (ptr->joinable()) {
            ptr->join();
        }
        delete ptr;
    }
}

void calculate(const std::string &probe_path, const std::string &encrypted_directory, const std::string &result_directory) {

    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);

    std::ifstream indexes_ifs("../data/gallery_data/gallery_indexes.dat", std::ios::binary);
    size_t cluster_size = 0;
    indexes_ifs.read(reinterpret_cast<char*>(&cluster_size), sizeof(size_t));

    std::ifstream centroids_ifs("../data/gallery_data/gallery_indexes.dat");
    std::vector<float> centroids(cluster_size * DIMENSION);
    indexes_ifs.read(reinterpret_cast<char*>(centroids.data()), cluster_size * DIMENSION * sizeof(float));
    centroids_ifs.close();

    size_t cluster_per_gpu = cluster_size / gpu_count;
    if (cluster_size % gpu_count != 0) {
        cluster_per_gpu++;
    }

    size_t poly_modulus_degree = ConfigUtil.int64ValueForKey("poly_modulus_degree");
    size_t scale_power = ConfigUtil.int64ValueForKey("scale_power");
    double scale = pow(2.0, scale_power);

    sigma::KernelProvider::initialize();

    sigma::EncryptionParameters params(sigma::scheme_type::ckks);
    params.set_poly_modulus_degree(poly_modulus_degree);
    auto modulus_bit_sizes = ConfigUtil.intVectorValueForKey("modulus_bit_sizes");
    params.set_coeff_modulus(sigma::CoeffModulus::Create(poly_modulus_degree, modulus_bit_sizes));
    sigma::SIGMAContext context(params);

    std::string encrypted_c1_data_path = encrypted_directory;
    if (encrypted_directory.back() != '/') {
        encrypted_c1_data_path += "/";
    }
    encrypted_c1_data_path += "encrypted_c1.dat";
    std::ifstream c1_ifs(encrypted_c1_data_path, std::ios::binary);
    sigma::Ciphertext c1;
    c1.use_half_data() = true;
    // TODO: check load with context
    c1.load(context, c1_ifs);
    c1_ifs.close();

    gallery_data_cluster.resize(gpu_count);
    for (uint gpu_index = 0; gpu_index < gpu_count - 1; gpu_index++) {
        gallery_data_cluster[gpu_index].resize(cluster_per_gpu);
    }
    gallery_data_cluster[gpu_count - 1].resize(cluster_size - (cluster_per_gpu * (gpu_count - 1)));
    indexes.resize(cluster_size);
    for (uint cluster_idx = 0; cluster_idx < cluster_size; cluster_idx++) {
        size_t indexes_size = 0;
        indexes_ifs.read(reinterpret_cast<char*>(&indexes_size), sizeof(size_t));
        indexes[cluster_idx].resize(indexes_size);
        indexes_ifs.read(reinterpret_cast<char*>(centroids.data()), indexes_size * sizeof(int64_t));

        auto gpu_idx = cluster_idx / cluster_per_gpu;
        auto idx = cluster_idx % cluster_per_gpu;

        std::ifstream cluster_ifs(gallery_data_path(encrypted_directory, cluster_idx), std::ios::binary);
        auto &cluster = gallery_data_cluster[gpu_idx][idx];
        cluster.resize(indexes_size);
        for (auto &ciphertext : cluster) {
            ciphertext.use_half_data() = true;
            ciphertext.load(context, cluster_ifs);
        }
    }

    probe_data = util::read_npy_data(probe_path);


//    c1.copy_to_device();
//
//    std::thread *threads[THREAD_SIZE];
//    for (auto &ptr: threads) {
//        ptr = new std::thread(calculate, std::ref(context), std::ref(c1), std::ref(scale));
//    }
//
//    for (auto &ptr: threads) {
//        if (ptr->joinable()) {
//            ptr->join();
//        }
//        delete ptr;
//    }

    gallery_data.clear();
    probe_data.clear();

}