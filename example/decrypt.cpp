
#include "../decrypt.h"

const std::string secret_key_data_path = "../data/secret_key.dat";
const std::string results_data_path = "../data/ip_results/top_ip_results.json";

int main() {

    decrypt(secret_key_data_path, results_data_path);

    return 0;
}
