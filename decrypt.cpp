//
// Created by scwang on 2024/3/6.
//

#include "decrypt.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <sigma.h>
#include <queue>

#include "extern/jsoncpp/json/json.h"
#include "util/configmanager.h"
#include "util/vectorutil.h"
#include "util/keyutil.h"

using namespace std;

std::string ip_results_path(size_t index) {
    return "../data/ip_results/probe_" + std::to_string(index) + "_results.dat";
}

class TopNPairs {

private:

    size_t n_;
    std::priority_queue<std::pair<double, size_t>, std::vector<std::pair<double, size_t>>, std::greater<>> pq_;

public:

    explicit TopNPairs(size_t n) : n_(n) {}

    void add(const std::pair<double, size_t> &value) {
        if (pq_.size() < n_) {
            pq_.push(value);
        } else {
            if (value.first > pq_.top().first) {
                pq_.pop();
                pq_.push(value);
            }
        }
    }

    std::vector<std::pair<double, size_t>> getData() {
        std::vector<std::pair<double, size_t>> results;
        while (!pq_.empty()) {
            results.push_back(pq_.top());
            pq_.pop();
        }
        std::reverse(results.begin(), results.end());
        return results;
    }

};

void load_results(vector<vector<sigma::Ciphertext>> &results, const sigma::SIGMAContext &context, std::ifstream &ifs) {
    size_t size1 = 0;
    ifs.read(reinterpret_cast<char *>(&size1), sizeof(size_t));
    results.resize(size1);
    for (auto &result : results) {
        size_t size2 = 0;
        ifs.read(reinterpret_cast<char *>(&size2), sizeof(size_t));
        result.resize(size2);
        for (auto &result1: result) {
            result1.use_half_data() = true;
            result1.load(context, ifs);
        }
    }
}

void load_cluster_indexes(vector<size_t> &cluster_indexes, std::ifstream &ifs) {
    size_t size = 0;
    ifs.read(reinterpret_cast<char *>(&size), sizeof(size_t));
    cluster_indexes.resize(size);
    ifs.read(reinterpret_cast<char *>(cluster_indexes.data()), size * sizeof(size_t));
}

int decrypt(const std::string &secret_key_path, const std::string &results_path) {

    size_t poly_modulus_degree = ConfigUtil.int64ValueForKey("poly_modulus_degree");
    size_t scale_power = ConfigUtil.int64ValueForKey("scale_power");
    double scale = pow(2.0, scale_power);

    sigma::KernelProvider::initialize();

    sigma::EncryptionParameters params(sigma::scheme_type::ckks);
    params.set_poly_modulus_degree(poly_modulus_degree);
    auto modulus_bit_sizes = ConfigUtil.intVectorValueForKey("modulus_bit_sizes");
    params.set_coeff_modulus(sigma::CoeffModulus::Create(poly_modulus_degree, modulus_bit_sizes));
//    params.setup_device_params(); // 初始化device相关参数
    sigma::SIGMAContext context(params);
//    context.setup_device_params(); // 初始化device相关参数

    sigma::CKKSEncoder encoder(context);

    sigma::SecretKey secret_key;
    util::load_secret_key(context, secret_key, secret_key_path);
    sigma::Decryptor decryptor(context, secret_key);

    size_t customized_scale_power = ConfigUtil.int64ValueForKey("customized_scale_power");
    double customized_scale = pow(2.0, customized_scale_power);


    std::ifstream indexes_ifs("../data/gallery_data/gallery_indexes.dat", std::ios::binary);
    size_t cluster_size = 0;
    indexes_ifs.read(reinterpret_cast<char*>(&cluster_size), sizeof(size_t));

    vector<vector<int64_t>> indexes(cluster_size);
    for (uint cluster_idx = 0; cluster_idx < cluster_size; cluster_idx++) {
        size_t indexes_size = 0;
        indexes_ifs.read(reinterpret_cast<char *>(&indexes_size), sizeof(size_t));
        indexes[cluster_idx].resize(indexes_size);
        indexes_ifs.read(reinterpret_cast<char *>(indexes[cluster_idx].data()), indexes_size * sizeof(int64_t));
    }


    Json::Value root;
    for (size_t i = 0;; i++) {
        std::ifstream ifs(ip_results_path(i), std::ios::binary);
        if (!ifs.good()) {
            break;
        }

        sigma::Ciphertext c1;
        c1.use_half_data() = true;
        c1.load(context, ifs);

        vector<vector<sigma::Ciphertext>> results;
        load_results(results, context, ifs);
        vector<size_t> cluster_indexes;
        load_cluster_indexes(cluster_indexes, ifs);
        sigma::Plaintext plaintext;
        std::vector<double> dest;
        TopNPairs pairs(5);
        for (int j = 0; j < results.size(); j++) {
            vector<sigma::Ciphertext> &cluster_result = results[j];
            auto probe_cluster_indexes = indexes[cluster_indexes[j]];
            size_t idx = 0;
            for (auto &ct : cluster_result) {
                decryptor.ckks_decrypt(ct, c1, plaintext);
                encoder.decode(plaintext, dest);
                for (auto ip : dest) {
                    if (idx >= probe_cluster_indexes.size()) {
                        break;
                    }
                    auto aa = ip / customized_scale;
                    auto bb = probe_cluster_indexes[idx];
                    auto pair = std::pair(aa, bb);
                    pairs.add(pair);
                    idx++;
                }
            }
        }
        ifs.close();
        Json::Value ips;
        auto data = pairs.getData();
        for (auto pair: data) {
            Json::Value pairValue;
            pairValue["inner_product"] = pair.first;
            pairValue["index"] = pair.second;
            ips.append(pairValue);
        }
        root.append(ips);
    }

    std::ofstream outputFile(results_path);
    if (outputFile.is_open()) {
        Json::StreamWriterBuilder writerBuilder;
        writerBuilder["indentation"] = "    ";
        std::unique_ptr<Json::StreamWriter> writer(writerBuilder.newStreamWriter());
        writer->write(root, &outputFile);
        outputFile.close();
        std::cout << "Data successfully written to disk." << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }

    return 0;
}
