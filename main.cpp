
#include "sigma/kernelprovider.h"
#include "utils.h"

#include <sigma/sigma.h>
#include <iostream>
#include <string>

const static std::string FILE_STORE_PATH = "../vectors/";
const static int GALLERY_SIZE = 1;

int main() {

    sigma::KernelProvider::initialize();

    auto gallery_data = read_npy_data(FILE_STORE_PATH + "gallery_x.npy");
    gallery_data.assign(gallery_data.begin(), gallery_data.begin() + GALLERY_SIZE);

    size_t poly_modulus_degree = 8192;
    double scale = pow(2.0, 40);

    sigma::EncryptionParameters params(sigma::scheme_type::ckks);
    params.set_poly_modulus_degree(poly_modulus_degree);
    params.set_coeff_modulus(sigma::CoeffModulus::BFVDefault(poly_modulus_degree));
    sigma::SIGMAContext context(params);
    sigma::CKKSEncoder encoder(context);

    std::cout << "GPU Encode begins" << std::endl;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < gallery_data.size(); i++) {
        sigma::Plaintext plain_vec;
        encoder.encode(gallery_data[i], scale, plain_vec);
    }
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
    std::cout << "GPU Encode ends " << gpu_duration.count() / 1000.f << "s" << std::endl;



    return 0;
}
