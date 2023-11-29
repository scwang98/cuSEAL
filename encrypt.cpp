
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

    std::cout << "Encode and encrypt start" << std::endl;
    auto time_start = std::chrono::high_resolution_clock::now();

    size_t poly_modulus_degree = ConfigUtil.int64ValueForKey("poly_modulus_degree");
    size_t scale_power = ConfigUtil.int64ValueForKey("scale_power");
    double scale = pow(2.0, scale_power);
    size_t customized_scale_power = ConfigUtil.int64ValueForKey("customized_scale_power");
    double customized_scale = pow(2.0, customized_scale_power);


    auto slots = poly_modulus_degree / 2;

    size_t gallery_size = 0;
    auto gallery_ptr = util::read_formatted_npy_data(FILE_STORE_PATH + "gallery_x.npy", slots, customized_scale, gallery_size);

    sigma::KernelProvider::initialize();

    sigma::EncryptionParameters params(sigma::scheme_type::ckks);
    params.set_poly_modulus_degree(poly_modulus_degree);
    auto modulus_bit_sizes = ConfigUtil.intVectorValueForKey("modulus_bit_sizes");
    params.set_coeff_modulus(sigma::CoeffModulus::Create(poly_modulus_degree, modulus_bit_sizes));
//    params.setup_device_params(); // 初始化device相关参数
    sigma::SIGMAContext context(params);
//    context.setup_device_params(); // 初始化device相关参数

    sigma::PublicKey public_key;
    sigma::SecretKey secret_key;
    util::load_public_key(context, public_key, public_key_data_path);
    util::load_secret_key(context, secret_key, secret_key_data_path);

    sigma::CKKSEncoder encoder(context);
//    sigma::Encryptor encryptor(context, public_key);
    sigma::Encryptor encryptor(context, secret_key);

    std::ofstream ofs(encrypted_data_path, std::ios::binary);

    for (int i = 0; i < gallery_size; ++i) {
        auto vec = gallery_ptr + (i * slots);
        sigma::Plaintext plain_vec;
        encoder.encode(vec, slots, scale, plain_vec);
        sigma::Ciphertext ciphertext;
        encryptor.encrypt(plain_vec, ciphertext);
        ciphertext.save(ofs);
        std::cout << "encrypt end " << i << std::endl;  // TODO: remove @wangshuchao
    }

    auto time_end = std::chrono::high_resolution_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
    std::cout << "Encode and encrypt end [" << time_diff.count() << " milliseconds]" << std::endl;

    ofs.close();

    delete[] gallery_ptr;

    return 0;
}
