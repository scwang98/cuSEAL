
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sigma.h>
#include <queue>

#include <faiss/IndexFlat.h>
#include <faiss/Clustering.h>

#include "../extern/jsoncpp/json/json.h"
#include "../util/configmanager.h"
#include "../util/vectorutil.h"
#include "../util/keyutil.h"

#define DIMENSION 512

using namespace std;

const std::string secret_key_data_path = "../data/secret_key.dat";
const std::string results_data_path = "../data/ip_results/top_ip_results_plaintext.json";

std::string gallery_data_path(size_t index) {
    std::ostringstream oss;
    oss << std::setw(5) << std::setfill('0') << index;
    return "../data/gallery_data/gallery_" + oss.str() + "_results.dat";
}

struct IPIndex {
    float inner_product;
    size_t index;

    IPIndex(float inner_product, size_t index) : inner_product(inner_product), index(index) {}

    bool operator<(const IPIndex &other) const {
        return inner_product < other.inner_product;
    }

    bool operator>(const IPIndex &other) const {
        return inner_product > other.inner_product;
    }
};

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

int main() {

    const std::string &secret_key_path = secret_key_data_path;
    const std::string gallery_path = "../vectors/gallery_x.npy";
    const std::string encrypted_directory = "../data";

    size_t poly_modulus_degree = ConfigUtil.int64ValueForKey("poly_modulus_degree");
    size_t scale_power = ConfigUtil.int64ValueForKey("scale_power");
    double scale = pow(2.0, scale_power);
    size_t customized_scale_power = ConfigUtil.int64ValueForKey("customized_scale_power");
    float customized_scale = pow(2.0, float(customized_scale_power));
    size_t nprobe = ConfigUtil.int64ValueForKey("nprobe");

    auto slots = poly_modulus_degree / 2;

    std::vector<std::vector<int64_t>> indexes;
    std::vector<float> centroids;
    size_t nlist = ConfigUtil.int64ValueForKey("nlist");
    auto tuples = util::read_cluster_npy_data(gallery_path, slots, customized_scale, nlist, centroids, indexes);

    std::vector<std::vector<float>> probe_data = util::read_npy_data("../vectors/probe_x.npy");
    auto probe_max_size = ConfigUtil.int64ValueForKey("probe_max_size");
    if (probe_max_size > 0 && probe_data.size() > probe_max_size) {
        probe_data = vector<vector<float>>(probe_data.begin(), probe_data.begin() + probe_max_size);
    }

    Json::Value root;
    for (auto &data: probe_data) {
        auto size = centroids.size() / DIMENSION;
        std::priority_queue<IPIndex, vector<IPIndex>, greater<>> pq;
        for (int i = 0; i < size; ++i) {
            auto start = centroids.data() + DIMENSION * i;
            float ip = 0;
            for (int j = 0; j < DIMENSION; ++j) {
                ip += (*(start + j) * data[j]);
            }
            pq.emplace(ip, i);
            if (i >= nprobe) {
                pq.pop();
            }
        }
        std::vector<size_t> clu_indexes;
        auto pq_size = pq.size();
        for (int i = 0; i < pq_size; ++i) {
            clu_indexes.push_back(pq.top().index);
            pq.pop();
        }


        TopNPairs pairs(5);
        for (auto idx: clu_indexes) {
            auto &tuple = tuples[idx];
            auto gallery_ptr = std::get<0>(tuple);
            auto gallery_size = std::get<1>(tuple);
            auto origin_size = std::get<2>(tuple);

            auto &index = indexes[idx];

            for (int k = 0; k < gallery_size / DIMENSION; k++) {
                auto dd_start = gallery_ptr + k * 4096 * 512;
                auto max_size = origin_size - k * 4096 < 4096 ? origin_size - k * 4096 : 4096;
                for (int a = 0; a < max_size; a++) {
                    auto start = dd_start + a;
                    float ip = 0;
                    for (int j = 0; j < DIMENSION; ++j) {
                        ip += (*(start + j * 4096) * data[j]);
                    }
                    pairs.add(make_pair(ip, index[a]));
                }
            }
        }
        Json::Value ips;
        for (auto pair: pairs.getData()) {
            Json::Value pairValue;
            pairValue["inner_product"] = pair.first;
            pairValue["index"] = pair.second;
            ips.append(pairValue);
        }
        root.append(ips);
    }
    std::ofstream outputFile(results_data_path);
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
