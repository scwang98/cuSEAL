
#include <iostream>
#include <fstream>
#include <sigma/sigma.h>

#include "extern/jsoncpp/json/json.h"
#include "util/configmanager.h"
#include "util/vectorutil.h"
#include "util/keyutil.h"

using namespace sigma;
using namespace std;

const static std::string FILE_STORE_PATH = "../vectors/";
const static int GALLERY_SIZE = 1;

int main() {

    TIMER_START;·

    size_t size = 8192;

    vector<vector<int64_t>> database(size, vector<int64_t>(size, 0));
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
//            database[i][j] = i * size + j;
            database[i][j] = j;
        }
    }

    for (int i = 1; i < size; ++i) {
        auto &arr = database[i];
        std::rotate(arr.begin(), arr.begin() + i, arr.end());
    }

    KernelProvider::initialize();

    EncryptionParameters params(sigma::scheme_type::bfv);
    size_t poly_modulus_degree = size;
    params.set_poly_modulus_degree(poly_modulus_degree);
    params.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
    params.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));

    sigma::SIGMAContext context(params);
    KeyGenerator keygen(context);

    sigma::PublicKey public_key;
    keygen.create_public_key(public_key);

    sigma::SecretKey secret_key = keygen.secret_key();

    GaloisKeys galois_keys;
    keygen.create_galois_keys(galois_keys);

    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    BatchEncoder batch_encoder(context);

    // Encrypting vector of zeros
    vector<int64_t> b_values(size, 0);
    b_values[3] = 1;
    Plaintext zeros;
    batch_encoder.encode(b_values, zeros);
    Ciphertext encrypted_zeros;
    encryptor.encrypt(zeros, encrypted_zeros);

    vector<Plaintext> encoded_database(size);
    for (int i = 0; i < size; ++i) {
        batch_encoder.encode(database[i], encoded_database[i]);
        cout << "database encode " << i << " end" << endl;
    }

    Ciphertext result;
    evaluator.multiply_plain(encrypted_zeros, encoded_database[0], result);
    for (int i = 1; i < size; ++i) {
//        Ciphertext rotated_zeros;
//        evaluator.rotate_rows(encrypted_zeros, i, galois_keys, rotated_zeros);
//        evaluator.multiply_plain_inplace(rotated_zeros, encoded_database[i]);
//        evaluator.add_inplace(result, rotated_zeros);
//        cout << "calculate " << i << " end" << endl;

        evaluator.rotate_rows_inplace(encrypted_zeros, 1, galois_keys);
        Ciphertext ciphertext;
        evaluator.multiply_plain(encrypted_zeros, encoded_database[i], ciphertext);
        evaluator.add_inplace(result, encrypted_zeros);
        cout << "calculate " << i << " end" << endl;
    }

    Plaintext plain_result;
    decryptor.decrypt(result, plain_result);
    std::vector<int64_t> decrypted_values;
    batch_encoder.decode(plain_result, decrypted_values);
    for (int i = 0; i < size; ++i) {
        std::cout << decrypted_values[i] << " ";
    }
    std::cout << std::endl;

    TIMER_PRINT_NOW(Calculate);

    return 0;
}
