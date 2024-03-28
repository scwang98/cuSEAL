
#include <string>
#include "../encrypt.h"

const std::string secret_key_data_path = "../data/secret_key.dat";

int main() {

    encrypt(secret_key_data_path, "../vectors/gallery_x.npy", "../data");

    return 0;
}
