
#include <iostream>
#include <fstream>
#include <vector>
#include <sigma/sigma.h>

#include "extern/jsoncpp/json/json.h"
#include "util/configmanager.h"
#include "util/vectorutil.h"
#include "util/keyutil.h"

const std::string encrypted_data_path = "../data/gallery.dat";
const std::string encrypted_c1_data_path = "../data/encrypted_c1.dat";
const std::string c1s_data_path = "../data/ip_results/encrypted_c1.dat";
const static std::string FILE_STORE_PATH = "../vectors/";

std::string ip_results_path(size_t index) {
    return "../data/ip_results/probe_" + std::to_string(index) + "_results.dat";
}

#define CUDA_TIME_START cudaEvent_t start, stop;\
                        cudaEventCreate(&start);\
                        cudaEventCreate(&stop);\
                        cudaEventRecord(start);\
                        cudaEventQuery(start);

#define CUDA_TIME_STOP cudaEventRecord(stop);\
                       cudaEventSynchronize(stop);\
                       float elapsed_time;\
                       cudaEventElapsedTime(&elapsed_time, start, stop);\
                       std::cout << "Time = " << elapsed_time << " ms." << std::endl;\

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

    std::ifstream c1_ifs(encrypted_c1_data_path, std::ios::binary);
    sigma::Ciphertext c1;
    c1.use_half_data() = true;
    // TODO: check load with context
    c1.load(context, c1_ifs);
    c1_ifs.close();

    std::vector<sigma::Ciphertext> gallery_data;
    std::ifstream gifs(encrypted_data_path, std::ios::binary);
    while (!gifs.eof()) {
        sigma::Ciphertext encrypted_vec;
        encrypted_vec.use_half_data() = true;
        try {
            encrypted_vec.load(context, gifs);
            gallery_data.push_back(encrypted_vec);
        } catch (const std::exception &e) {
            break;
        }
    }
    gifs.close();

    auto probe_data = util::read_npy_data(FILE_STORE_PATH + "probe_x.npy");
    // TODO: remove @wangshuchao
    probe_data.assign(probe_data.begin(), probe_data.begin() + 100);

    sigma::CKKSEncoder encoder(context);
    sigma::Evaluator evaluator(context);

    size_t dimension = 512;

    std::ofstream c1_ofs(c1s_data_path, std::ios::binary);

    TIMER_START;

    c1.copy_to_device();

    auto c1_sum = c1;
    auto c1_row = c1;
    sigma::Ciphertext result;
    sigma::Ciphertext row;

    std::vector<sigma::Plaintext> encoded_probes(dimension);
    for (size_t pi = 0; pi < probe_data.size(); ++pi) {
        const auto& probe = probe_data[pi];
//        std::vector<sigma::Plaintext> encoded_probes(dimension);
        // 0.012
        encoder.encode(probe[0], scale, encoded_probes[0]);
        // 0.035
        encoded_probes[0].copy_to_device();
        // 0.029
        c1_sum.copy_on_device(c1);
        // 0.011
        evaluator.my_multiply_plain_inplace(c1_sum, encoded_probes[0]);
        for (int i = 1; i < dimension; ++i) {

            // 0.008
            encoder.encode(probe[i], scale, encoded_probes[i]);
            // 0.029
            encoded_probes[i].copy_to_device();

            // 0.019
            c1_row.copy_on_device(c1);
            // 0.008
            evaluator.my_multiply_plain_inplace(c1_row, encoded_probes[i]);
            // 0.006
            evaluator.my_add_inplace(c1_sum, c1_row);
        }
        // 0.07
        c1_sum.save(c1_ofs);

        std::ofstream ofs(ip_results_path(pi), std::ios::binary);
        size_t calculate_size = gallery_data.size() / 512 * 512;
        std::vector<sigma::Ciphertext> results;
        for (size_t offset = 0; offset < calculate_size; offset += dimension) {
            // TODO: gallery_data中的数据一次复制完成
            if (pi == 0) {
                gallery_data[offset].copy_to_device();
            }
            // 0.022
            result.copy_on_device(gallery_data[offset]);
            // 0.008
            evaluator.my_multiply_plain_inplace(result, encoded_probes[0]);
            for (size_t i = 1; i < dimension; i++) {
                if (pi == 0) {
                    gallery_data[offset + i].copy_to_device();
                }
                // 0.022
                row.copy_on_device(gallery_data[offset + i]);

                // 0.007
                evaluator.my_multiply_plain_inplace(row, encoded_probes[i]);

                // 0.007
                evaluator.my_add_inplace(result, row);

//                std::cout << "offset=" << offset << " " << "i=" << i << std::endl;
            }
            // 0.041
            result.retrieve_to_host();
            CUDA_TIME_START
            // 0.065
            result.save(ofs);
            CUDA_TIME_STOP
        }
        ofs.close();
//        std::cout << "calculate end " << pi << std::endl;  // TODO: remove @wangshuchao
    }
    c1_ofs.close();

    TIMER_PRINT_NOW(Calculate_inner_product);

    return 0;
}
