
#include "../keygen.h"

const std::string secret_key_data_path = "../data/secret_key.dat";

int main() {

    keygen(secret_key_data_path);

    return 0;
}
