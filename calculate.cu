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

public:

    std::vector<float> probe_data;
    std::vector<size_t> indexes;
    size_t finished_part;
    std::vector<sigma::Ciphertext> c1_sums;
    vector<vector<vector<sigma::Ciphertext>>> results;
    vector<vector<vector<size_t>>> result_indexes;

    Task() = default;

    Task(const std::vector<float>& data, std::vector<size_t> &indexes, int gpu_count) : probe_data(data), indexes(indexes) {
        finished_part = 0;
        c1_sums.resize(gpu_count);
        results.resize(gpu_count);
    }

};

struct IPIndex {
    float inner_product;
    size_t index;

    IPIndex(float inner_product, size_t index) : inner_product(inner_product), index(index) {}

    bool operator<(const IPIndex& other) const {
        return inner_product > other.inner_product;
    }
};

class TaskManager {

    std::vector<float> centroids;

public:
    int gpu_count;
    unordered_set<Task *> set;
    vector<util::safe_queue<Task *, 100>> queues;

    util::safe_queue<Task *, 400> finished_queue;

    TaskManager(int gpu_count, std::vector<float> &centroids) : gpu_count(gpu_count), centroids(centroids) {
        queues.resize(gpu_count);
    }

    Task *start_task(const std::vector<float>& data) {
        auto size = centroids.size() / DIMENSION;
        std::priority_queue<IPIndex> pq;
        for (int i = 0; i < size; ++i) {
            auto start = centroids.data() + DIMENSION * i;
            float ip = 0;
            for (int j = 0; j < DIMENSION; ++j) {
                ip += (*(start + j) * data[j]);
            }
            pq.push(IPIndex(ip, i));
            if (i >= 5) {
                pq.pop();
            }
        }
        std::vector<size_t> indexes;
        for (int i = 0; i < pq.size(); ++i) {
            indexes.push_back(pq.top().index);
            pq.pop();
        }

        auto task = new Task(data, indexes, gpu_count);
        set.insert(task);
        for (uint i = 0; i < gpu_count; i++) {
            queues[i].push(task);
        }
        return task;
    }

    void probe_finished() {
        auto task = new Task();
        for (uint i = 0; i < gpu_count; i++) {
            queues[i].push(task);
        }
    }

    void task_finished(Task *task) {
        finished_queue.push(task);
    }
};

//std::vector<sigma::Ciphertext> gallery_data;
vector<vector<vector<sigma::Ciphertext>>> gallery_data_cluster;
std::vector<std::vector<int64_t>> indexes;
std::vector<std::vector<float>> probe_data;

TaskManager *task_manager;

void save_thread() {
//    while (true) {
//        if (finished) {
//            break;
//        }
//        auto task = task_manager->finished_queue.pop();
//        task->finished_part++;
//        if (task->finished_part >= task_manager->gpu_count) {
//            // TODO: 数据存储
//
//            task_manager->set.erase(task);
//            delete task;
//        }
//    }
}

void calculate_thread(int gpu_index, int cluster_per_gpu, sigma::SIGMAContext &context, const sigma::Ciphertext &c1, double scale) {
    cudaSetDevice(gpu_index);

    auto gpu_gallery_data = gallery_data_cluster[gpu_index];

    const uint cluster_index_start = cluster_per_gpu * gpu_index;
    const uint cluster_index_end = cluster_index_start + gpu_gallery_data.size();

    sigma::CKKSEncoder encoder(context);
    sigma::Evaluator evaluator(context);

    sigma::Ciphertext c1_sum;
    sigma::Ciphertext c1_row;
    sigma::Ciphertext result;
    sigma::Ciphertext row;

    std::vector<sigma::Plaintext> encoded_probes(DIMENSION);

    while (true) {

        auto task = task_manager->queues[gpu_index].pop();

        const auto &probe = task->probe_data;

        if (probe.empty()) {

            break;
        }

        encoder.cu_encode(probe[0], scale, encoded_probes[0]);

        evaluator.cu_multiply_plain(c1, encoded_probes[0], c1_sum);
        for (int i = 1; i < DIMENSION; ++i) {

            encoder.cu_encode(probe[i], scale, encoded_probes[i]);

            evaluator.cu_multiply_plain(c1, encoded_probes[i], c1_row);

            evaluator.cu_add_inplace(c1_sum, c1_row);
        }

        c1_sum.retrieve_to_host();

//        std::ofstream ofs(ip_results_path(result_directory, index), std::ios::binary);
//        c1_sum.save(ofs);
        task->c1_sums[gpu_index].copy_from(c1_sum, false);

        vector<size_t> filtered_indexes;
        for (auto index : task->indexes) {
            if (cluster_index_start <= index && index < cluster_index_end) {
                filtered_indexes.push_back(index - cluster_index_start);
            }
        }

        task->results[gpu_index].resize(gpu_gallery_data.size());
        task->result_indexes[gpu_index].resize(gpu_gallery_data.size());
        for (int cluster_index = 0; cluster_index < gpu_gallery_data.size(); cluster_index++) {
            auto gallery_data = gpu_gallery_data[cluster_index];
            for (size_t offset = 0; offset < gallery_data.size(); offset += DIMENSION) {

                evaluator.cu_multiply_plain(gallery_data[offset], encoded_probes[0], result);
                for (size_t i = 1; i < DIMENSION; i++) {

                    evaluator.cu_multiply_plain(gallery_data[offset + i], encoded_probes[i], row);

                    evaluator.cu_add_inplace(result, row);
                }

                result.retrieve_to_host();
//            result.save(ofs);
                task->results[gpu_index][cluster_index].emplace_back(result, false);
            }
        }

        task_manager->task_finished(task);


//        ofs.close();
    }
}

void task_for_gpu(int gpu_index, int cluster_per_gpu, sigma::SIGMAContext &context, const sigma::Ciphertext &origin_c1, double scale) {
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
        ptr = new std::thread(calculate_thread, gpu_index, cluster_per_gpu, std::ref(context), std::ref(c1), std::ref(scale));
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

    std::ifstream centroids_ifs("../data/gallery_data/gallery_centroids.dat");
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

    task_manager = new TaskManager(gpu_count, centroids);

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
        indexes_ifs.read(reinterpret_cast<char*>(indexes[cluster_idx].data()), indexes_size * sizeof(int64_t));

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

    std::thread *threads[gpu_count];
    for (int i = 0; i < gpu_count; i++) {
        threads[i] = new std::thread(task_for_gpu, i, cluster_per_gpu, std::ref(context), std::ref(c1), scale);
    }

    probe_data = util::read_npy_data(probe_path);


    for (auto data : probe_data) {
        task_manager->start_task(data);
    }

    task_manager->probe_finished();

    for (auto &ptr: threads) {
        if (ptr->joinable()) {
            ptr->join();
        }
        delete ptr;
    }

    probe_data.clear();

}