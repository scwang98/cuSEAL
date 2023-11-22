
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

void merge_slots(sigma::Evaluator &evaluator, sigma::Ciphertext ciphertext, const sigma::GaloisKeys& galois_keys) {
    auto slots = ciphertext.poly_modulus_degree();
    sigma::Ciphertext ciphertext2;
    int steps = 1;
    for (;slots > 0; slots >> 1, steps << 1) {
        evaluator.rotate_vector(ciphertext, steps, galois_keys, ciphertext2);
        evaluator.add_inplace(ciphertext, ciphertext2);
    }
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
    gallery_data.assign(gallery_data.begin(), gallery_data.begin() + 4096);

    auto probe_data = util::read_npy_data(FILE_STORE_PATH + "probe_x.npy");
    // TODO: remove @wangshuchao
    probe_data.assign(probe_data.begin(), probe_data.begin() + 1);

    sigma::CKKSEncoder encoder(context);
    sigma::Evaluator evaluator(context);

//    sigma::GaloisKeys galois_keys;
//    util::load_galois_key(context, galois_keys, galois_keys_data_path);

    size_t dimension = 512;
    auto slots = poly_modulus_degree / 2;
    auto batch_size = slots / dimension;

    std::vector<std::vector<sigma::Ciphertext>> final_results;
    for (const auto& probe: probe_data) {
        std::vector<sigma::Plaintext> encoded_probes(dimension);
        for (int i = 0; i < dimension; ++i) {
//            std::vector<double> p(slots, probe[i]);
//            encoder.encode(p, scale, encoded_probes[i]);
            encoder.encode(probe[i], scale, encoded_probes[i]);
        }

        std::vector<sigma::Ciphertext> results;
        for (size_t offset = 0; offset < gallery_data.size(); offset += dimension) {
            for (size_t i = 0; i < dimension; i++) {
                evaluator.multiply_plain_inplace(gallery_data[offset + i], encoded_probes[i]);
                if (i > 0) {
                    evaluator.add_inplace(gallery_data[offset], gallery_data[offset + i]);
                }
                std::cout << "offset=" << offset << " " << "i=" << i << std::endl;
            }
            results.push_back(gallery_data[offset]);
        }
        final_results.push_back(results);
    }

    std::cout << final_results[0][0].data()[0] << std::endl;

    sigma::SecretKey secret_key;
    const std::string secret_key_data_path = "../data/secret_key.dat";
    util::load_secret_key(context, secret_key, secret_key_data_path);
    sigma::Decryptor decryptor(context, secret_key);
    size_t customized_scale_power = ConfigUtil.int64ValueForKey("customized_scale_power");
    double customized_scale = pow(2.0, customized_scale_power);
    sigma::Plaintext rrr;
    decryptor.decrypt(final_results[0][0], rrr);
    std::vector<double> dest;
    encoder.decode(rrr, dest);

    for (int i = 0; i < 8; ++i) {
        std::cout << dest[i] / customized_scale << std::endl;
    }

    return 0;
}
