
#include <iostream>
#include <fstream>
#include <sigma/sigma.h>

#include "extern/jsoncpp/json/json.h"
#include "util/configmanager.h"
#include "util/vectorutil.h"
#include "util/keyutil.h"

const static std::string FILE_STORE_PATH = "../vectors/";
const static int GALLERY_SIZE = 1;

int main() {

//    sigma::KernelProvider::initialize();
//
//    auto gallery_data = util::read_npy_data(FILE_STORE_PATH + "gallery_x.npy");
//    gallery_data.assign(gallery_data.begin(), gallery_data.begin() + GALLERY_SIZE);
//
//    size_t poly_modulus_degree = 8192;
//    double scale = pow(2.0, 40);
//
//    sigma::EncryptionParameters params(sigma::scheme_type::ckks);
//    params.set_poly_modulus_degree(poly_modulus_degree);
//    params.set_coeff_modulus(sigma::CoeffModulus::BFVDefault(poly_modulus_degree));
////    params.setup_device_params(); // 初始化device相关参数
//    sigma::SIGMAContext context(params);
////    context.setup_device_params(); // 初始化device相关参数
//    sigma::CKKSEncoder encoder(context);
//
//    std::cout << "GPU Encode begins" << std::endl;
//    auto gpu_start = std::chrono::high_resolution_clock::now();
//    for (size_t i = 0; i < gallery_data.size(); i++) {
//        sigma::Plaintext plain_vec;
//        encoder.encode(gallery_data[i], scale, plain_vec);
//        std::cout << "[" << i << "]=" << plain_vec[i] << std::endl;
//    }
//    auto gpu_end = std::chrono::high_resolution_clock::now();
//    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
//    std::cout << "GPU Encode ends " << gpu_duration.count() / 1000.f << "s" << std::endl;

    size_t poly_modulus_degree = ConfigUtil.int64ValueForKey("poly_modulus_degree");
    size_t scale_power = ConfigUtil.int64ValueForKey("scale_power");
    double scale = pow(2.0, scale_power);
    auto slots = poly_modulus_degree / 2;

    // TODO: remove @wangshuchao
    sigma::KernelProvider::initialize();

    sigma::EncryptionParameters parms(sigma::scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    auto modulus_bit_sizes = ConfigUtil.intVectorValueForKey("modulus_bit_sizes");
    parms.set_coeff_modulus(sigma::CoeffModulus::Create(poly_modulus_degree, modulus_bit_sizes));

    sigma::SIGMAContext context(parms);

    sigma::KeyGenerator keygen(context);
    sigma::SecretKey secret_key;
    secret_key = keygen.secret_key();

    double data[] = {12.456134613, 13.43123413461346, 42.2456154134, 13.546723562356};

    sigma::CKKSEncoder encoder(context);
    sigma::Encryptor encryptor(context, secret_key);

    sigma::Ciphertext c1;
    c1.use_half_data() = true;
    encryptor.sample_symmetric_ckks_c1(c1);

    sigma::Plaintext plain_vec;
    encoder.encode(data, 4, scale, plain_vec);
    sigma::Ciphertext ciphertext;
    ciphertext.use_half_data() = true;
    encryptor.encrypt_symmetric_ckks(plain_vec, ciphertext, c1);

    sigma::Plaintext plaintext;
    double data2[] = {0.35345, 0.1324514, 0.132451, 0.1346523146};
    encoder.encode(data2, 4, scale, plaintext);

    sigma::Evaluator evaluator(context);

    evaluator.multiply_plain_inplace(c1, plaintext);

    evaluator.multiply_plain_inplace(ciphertext, plaintext);

    sigma::Decryptor decryptor(context, secret_key);

    sigma::Plaintext result;
    decryptor.ckks_decrypt(ciphertext, c1, result);
    std::vector<double> dest;
    encoder.decode(result, dest);

    printf("%lf\n", dest[0]);
    printf("%lf\n", dest[1]);
    printf("%lf\n", dest[2]);
    printf("%lf\n", dest[3]);
//    std::cout << dest[0] << std::endl;
//    std::cout << dest[1] << std::endl;
//    std::cout << dest[2] << std::endl;
//    std::cout << dest[3] << std::endl;

    return 0;
}
