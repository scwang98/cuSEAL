
#include <iostream>
#include <fstream>
#include <sigma/sigma.h>

#include "extern/jsoncpp/json/json.h"
#include "util/configmanager.h"
#include "util/vectorutil.h"
#include "util/keyutil.h"

const std::string public_key_data_path = "../data/public_key.dat";
const std::string secret_key_data_path = "../data/secret_key.dat";
const std::string encrypted_data_path = "../data/gallery.dat";
const static std::string FILE_STORE_PATH = "../vectors/";

int main() {

//    auto gallery_data = util::read_npy_data(FILE_STORE_PATH + "gallery_x.npy");
//
//    gallery_data.assign(gallery_data.begin(), gallery_data.begin() + 32768); // TODO: remove @wangshuchao

    size_t poly_modulus_degree = util::ConfigManager::singleton.int64ValueForKey("poly_modulus_degree");
    size_t scale_power = util::ConfigManager::singleton.int64ValueForKey("scale_power");
    double scale = pow(2.0, scale_power);


    auto slots = poly_modulus_degree / 2;
    auto batch_size = slots / 512;

    auto gallery_data = util::batch_read_npy_data(FILE_STORE_PATH + "gallery_x.npy", batch_size);

    auto gallery_data_size = gallery_data.size();
    gallery_data.assign(gallery_data.begin(), gallery_data.begin() + (gallery_data_size - (gallery_data_size % slots)));

    for (size_t offset = 0; offset < gallery_data.size(); offset += slots) {
//        for (size_t i = 0; i < slots; i++) {
//            auto &vec = gallery_data[offset + i];
//            std::rotate(vec.begin(), vec.begin() + i, vec.end());
//        }
        for (size_t i = 0; i < slots; ++i) {
            for (size_t j = i + 1; j < slots; ++j) {
                std::swap(gallery_data[offset + i][j], gallery_data[offset + j][i]);
            }
        }
    }

    sigma::KernelProvider::initialize();

    sigma::EncryptionParameters params(sigma::scheme_type::ckks);
    params.set_poly_modulus_degree(poly_modulus_degree);
    params.set_coeff_modulus(sigma::CoeffModulus::BFVDefault(poly_modulus_degree));
//    params.setup_device_params(); // 初始化device相关参数
    sigma::SIGMAContext context(params);
//    context.setup_device_params(); // 初始化device相关参数

    sigma::PublicKey public_key;
    sigma::SecretKey secret_key;
    util::load_public_key(context, public_key, public_key_data_path);
    util::load_secret_key(context, secret_key, secret_key_data_path);

    sigma::CKKSEncoder encoder(context);
    sigma::Encryptor encryptor(context, public_key);

    std::ofstream ofs(encrypted_data_path, std::ios::binary);

    for (int i = 0; i < gallery_data.size(); ++i) {
//    for (int i = 0; i < 10; ++i) {
        auto &vec = gallery_data[i];
        sigma::Plaintext plain_vec;
        encoder.encode(vec, scale, plain_vec);
        sigma::Ciphertext ciphertext;
        encryptor.encrypt(plain_vec, ciphertext);
        ciphertext.save(ofs);
        std::cout << "encrypt end " << i << std::endl;
    }
//    for (auto &vec: gallery_data) {
//        sigma::Plaintext plain_vec;
//        encoder.encode(vec, scale, plain_vec);
//        sigma::Ciphertext ciphertext;
//        encryptor.encrypt(plain_vec, ciphertext);
//        ciphertext.save(ofs);
//    }

    return 0;
}
