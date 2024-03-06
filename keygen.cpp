//
// Created by scwang on 2024/3/6.
//

#include "keygen.h"
#include <fstream>
#include <sigma.h>

#include "extern/jsoncpp/json/json.h"
#include "util/configmanager.h"
#include "util/keyutil.h"

void keygen(const std::string &secret_key_path) {

    size_t poly_modulus_degree = ConfigUtil.int64ValueForKey("poly_modulus_degree");

    sigma::KernelProvider::initialize();

    sigma::EncryptionParameters params(sigma::scheme_type::ckks);
    params.set_poly_modulus_degree(poly_modulus_degree);
    auto modulus_bit_sizes = ConfigUtil.intVectorValueForKey("modulus_bit_sizes");
    params.set_coeff_modulus(sigma::CoeffModulus::Create(poly_modulus_degree, modulus_bit_sizes));

    sigma::SIGMAContext context(params);

    sigma::KeyGenerator keygen(context);

    sigma::SecretKey secret_key;
    secret_key = keygen.secret_key();
    util::save_secret_key(secret_key, secret_key_path);

}
