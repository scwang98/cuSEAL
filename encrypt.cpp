//
// Created by scwang on 2024/3/6.
//

#include "encrypt.h"
#include <iostream>
#include <fstream>
#include <sigma.h>
#include <string>

#include "extern/jsoncpp/json/json.h"
#include "util/configmanager.h"
#include "util/vectorutil.h"
#include "util/keyutil.h"

void
encrypt(const std::string &secret_key_path, const std::string &gallery_path, const std::string &encrypted_directory) {

    size_t poly_modulus_degree = ConfigUtil.int64ValueForKey("poly_modulus_degree");
    size_t scale_power = ConfigUtil.int64ValueForKey("scale_power");
    double scale = pow(2.0, scale_power);
    size_t customized_scale_power = ConfigUtil.int64ValueForKey("customized_scale_power");
    float customized_scale = pow(2.0, float(customized_scale_power));

    auto slots = poly_modulus_degree / 2;

    size_t gallery_size = 0;
    auto gallery_ptr = util::read_formatted_npy_data(gallery_path, slots, customized_scale, gallery_size);

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

    std::string encrypted_data_path = encrypted_directory;
    if (encrypted_directory.back() != '/') {
        encrypted_data_path += "/";
    }
    encrypted_data_path += "gallery.dat";
    std::ofstream ofs(encrypted_data_path, std::ios::binary);

    std::cout << "Encode and encrypt start" << std::endl;
    auto time_start = std::chrono::high_resolution_clock::now();

    sigma::Plaintext plain_vec;
    sigma::Ciphertext ciphertext;
    for (int i = 0; i < gallery_size; ++i) {
        auto vec = gallery_ptr + (i * slots);

        auto time_start0 = std::chrono::high_resolution_clock::now();

        encoder.encode_float(vec, slots, scale, plain_vec);

        ciphertext.use_half_data() = true;
        encryptor.encrypt_symmetric_ckks(plain_vec, ciphertext, c1);

        ciphertext.retrieve_to_host();

        ciphertext.save(ofs);
    }

    auto time_end = std::chrono::high_resolution_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
    std::cout << "Encode and encrypt end [" << time_diff.count() << " milliseconds]" << std::endl;

    c1.release_device_data();

    ofs.close();

    delete[] gallery_ptr;
}
