
#include <iostream>
#include <fstream>
#include <sigma/sigma.h>

#include "extern/jsoncpp/json/json.h"
#include "util/configmanager.h"
#include "util/keyutil.h"

const std::string public_key_data_path = "../data/public_key.dat";
const std::string secret_key_data_path = "../data/secret_key.dat";
const std::string galois_keys_data_path = "../data/galois_keys.dat";

bool fileExists(const std::string& filename) {
    std::ifstream file(filename.c_str());
    return file.good();
}

int main() {

//    if (fileExists(public_key_data_path) || fileExists(secret_key_data_path)) {
//        std::cout << "目录下存在密钥文件，是否覆盖？ [y/n]: ";
//        char response;
//        std::cin >> response;
//
//        if (response == 'n' || response == 'N') {
//            std::cout << "程序结束。\n";
//            return 0;
//        } else if (response != 'y' && response != 'Y') {
//            // 无效的输入，结束程序
//            std::cout << "无效的输入，程序结束。\n";
//            return 0;
//        }
//    }

    size_t poly_modulus_degree = ConfigUtil.int64ValueForKey("poly_modulus_degree");

    // TODO: remove @wangshuchao
    sigma::KernelProvider::initialize();

    sigma::EncryptionParameters parms(sigma::scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    auto modulus_bit_sizes = ConfigUtil.intVectorValueForKey("modulus_bit_sizes");
    parms.set_coeff_modulus(sigma::CoeffModulus::Create(poly_modulus_degree, modulus_bit_sizes));

    sigma::SIGMAContext context(parms);

    sigma::KeyGenerator keygen(context);

//    sigma::PublicKey public_key;
//    keygen.create_public_key(public_key);
//    util::save_public_key(public_key, public_key_data_path);

    sigma::SecretKey secret_key;
    secret_key = keygen.secret_key();
    util::save_secret_key(secret_key, secret_key_data_path);

//    sigma::GaloisKeys galois_keys;
//    keygen.create_galois_keys(galois_keys);
//    util::save_galois_keys(galois_keys, galois_keys_data_path);

    return 0;
}
