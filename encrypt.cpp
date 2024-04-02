//
// Created by scwang on 2024/3/6.
//

#include "encrypt.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sigma.h>
#include <string>

#include <faiss/IndexFlat.h>
#include <faiss/Clustering.h>

#include "extern/jsoncpp/json/json.h"
#include "util/configmanager.h"
#include "util/vectorutil.h"
#include "util/keyutil.h"

std::string gallery_data_path(size_t index) {
    std::ostringstream oss;
    oss << std::setw(5) << std::setfill('0') << index;
    return "../data/gallery_data/gallery_" + oss.str() + "_results.dat";
}

void
encrypt(const std::string &secret_key_path, const std::string &gallery_path, const std::string &encrypted_directory) {

    size_t poly_modulus_degree = ConfigUtil.int64ValueForKey("poly_modulus_degree");
    size_t scale_power = ConfigUtil.int64ValueForKey("scale_power");
    double scale = pow(2.0, scale_power);
    size_t customized_scale_power = ConfigUtil.int64ValueForKey("customized_scale_power");
    float customized_scale = pow(2.0, float(customized_scale_power));

    auto slots = poly_modulus_degree / 2;

//    size_t gallery_size = 0;
//    auto gallery_ptr = util::read_formatted_npy_data(FILE_STORE_PATH + "gallery_x.npy", slots, customized_scale, gallery_size);

    std::vector<std::vector<int64_t>> indexes;
    std::vector<float> centroids;
    auto tuples = util::read_cluster_npy_data(gallery_path, slots, customized_scale, 100, centroids, indexes);

    sigma::KernelProvider::initialize();

    sigma::EncryptionParameters params(sigma::scheme_type::ckks);
    params.set_poly_modulus_degree(poly_modulus_degree);
    auto modulus_bit_sizes = ConfigUtil.intVectorValueForKey("modulus_bit_sizes");
    params.set_coeff_modulus(sigma::CoeffModulus::Create(poly_modulus_degree, modulus_bit_sizes));
    sigma::SIGMAContext context(params);

    sigma::SecretKey secret_key;
    util::load_secret_key(context, secret_key, secret_key_path);

    secret_key.copy_to_device();

    sigma::CKKSEncoder encoder(context);
    sigma::Encryptor encryptor(context, secret_key);

    sigma::Ciphertext c1;
    c1.use_half_data() = true;
    encryptor.sample_symmetric_ckks_c1(c1);

    std::string encrypted_c1_data_path = encrypted_directory;
    if (encrypted_directory.back() != '/') {
        encrypted_c1_data_path += "/";
    }
    encrypted_c1_data_path += "encrypted_c1.dat";
    std::ofstream c1_ofs(encrypted_c1_data_path, std::ios::binary);
    c1.save(c1_ofs);
    c1_ofs.close();

    c1.copy_to_device();

    std::ofstream centroids_ofs(encrypted_directory + "/gallery_data/gallery_centroids.dat");
    centroids_ofs.write(reinterpret_cast<const char *>(centroids.data()), centroids.size() * sizeof(float));
    centroids_ofs.close();

    std::ofstream indexes_ofs(encrypted_directory + "/gallery_data/gallery_indexes.dat");
    auto indexes_size = indexes.size();
    indexes_ofs.write(reinterpret_cast<const char *>(&indexes_size), sizeof(size_t));

    std::cout << "Encode and encrypt start" << std::endl;
    auto time_start = std::chrono::high_resolution_clock::now();

    for (uint idx = 0; idx < tuples.size(); idx++) {
        auto &tuple = tuples[idx];
        auto gallery_ptr = std::get<0>(tuple);
        auto gallery_size = std::get<1>(tuple);
        auto origin_size = std::get<2>(tuple);
        auto index = indexes[idx];

        size_t dataSize = index.size();
        indexes_ofs.write(reinterpret_cast<const char *>(&dataSize), sizeof(size_t));
        indexes_ofs.write(reinterpret_cast<const char *>(index.data()), dataSize * sizeof(int64_t));

        auto path = gallery_data_path(idx);
        std::ofstream ofs(path, std::ios::binary);

        ofs.write(reinterpret_cast<const char *>(&gallery_size), sizeof(size_t));

        sigma::Plaintext plain_vec;
        sigma::Ciphertext ciphertext;
        for (int i = 0; i < gallery_size; ++i) {
            auto vec = gallery_ptr + (i * slots);

            encoder.encode_float(vec, slots, scale, plain_vec);

            ciphertext.use_half_data() = true;
            encryptor.encrypt_symmetric_ckks(plain_vec, ciphertext, c1);

            ciphertext.retrieve_to_host();

            ciphertext.save(ofs);
        }

        ofs.close();
        delete[] gallery_ptr;
    }

    auto time_end = std::chrono::high_resolution_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
    std::cout << "Encode and encrypt end [" << time_diff.count() << " milliseconds]" << std::endl;

    c1.release_device_data();

//    ofs.close();
    indexes_ofs.close();

//    delete[] gallery_ptr;

}
