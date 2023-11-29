
#include <iostream>
#include <fstream>
#include <vector>
#include <sigma/sigma.h>

#include "extern/jsoncpp/json/json.h"
#include "util/configmanager.h"
#include "util/vectorutil.h"
#include "util/keyutil.h"

const std::string galois_keys_data_path = "../data/galois_keys.dat";
const std::string encrypted_data_path = "../data/gallery.dat";
const static std::string FILE_STORE_PATH = "../vectors/";

std::string ip_results_path(size_t index) {
    return "../data/ip_results/probe_" + std::to_string(index) + "_results.dat";
}

int main() {

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

    std::vector<sigma::Ciphertext> gallery_data;
    std::ifstream gifs(encrypted_data_path, std::ios::binary);
    while (!gifs.eof()) {
        sigma::Ciphertext encrypted_vec;
        try {
            encrypted_vec.load(context, gifs);
            gallery_data.push_back(encrypted_vec);
        } catch (const std::exception &e) {
            break;
        }
    }
//    gallery_data.assign(gallery_data.begin(), gallery_data.begin() + 4096);

    auto probe_data = util::read_npy_data(FILE_STORE_PATH + "probe_x.npy");
    // TODO: remove @wangshuchao
    probe_data.assign(probe_data.begin(), probe_data.begin() + 2);

    sigma::CKKSEncoder encoder(context);
    sigma::Evaluator evaluator(context);

//    sigma::GaloisKeys galois_keys;
//    util::load_galois_key(context, galois_keys, galois_keys_data_path);

    size_t dimension = 512;
    auto slots = poly_modulus_degree / 2;
    auto batch_size = slots / dimension;

    for (size_t pi = 0; pi < probe_data.size(); ++pi) {
        const auto& probe = probe_data[pi];
        std::vector<sigma::Plaintext> encoded_probes(dimension);
        for (int i = 0; i < dimension; ++i) {
            encoder.encode(probe[i], scale, encoded_probes[i]);
        }

        std::ofstream ofs(ip_results_path(pi), std::ios::binary);
        size_t calculate_size = gallery_data.size() / 512 * 512;
        for (size_t offset = 0; offset < calculate_size; offset += dimension) {
            auto result = gallery_data[offset];
            evaluator.multiply_plain_inplace(result, encoded_probes[0]);
            for (size_t i = 1; i < dimension; i++) {
                auto row = gallery_data[offset + i];
                evaluator.multiply_plain_inplace(row, encoded_probes[i]);
                evaluator.add_inplace(result, row);
                std::cout << "offset=" << offset << " " << "i=" << i << std::endl;
            }
            result.save(ofs);
        }
        ofs.close();
        std::cout << "calculate end " << pi << std::endl;  // TODO: remove @wangshuchao
    }

    return 0;
}
