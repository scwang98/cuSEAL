
#include <iostream>
#include <fstream>
#include <queue>
#include <sigma/sigma.h>

#include "extern/jsoncpp/json/json.h"
#include "util/configmanager.h"
#include "util/vectorutil.h"
#include "util/keyutil.h"
#include "util/safequeue.h"

const std::string public_key_data_path = "../data/public_key.dat";
const std::string secret_key_data_path = "../data/secret_key.dat";
const std::string encrypted_data_path = "../data/gallery.dat";
const std::string encrypted_c1_data_path = "../data/encrypted_c1.dat";
const static std::string FILE_STORE_PATH = "../vectors/";
const size_t cuda_stream_num = 32;

util::safe_queue<sigma::util::HostGroup<float> *, 10> source_queue;
util::safe_queue<sigma::Ciphertext *> store_queue;

//bool source_data_read_finished = false;

//std::queue<std::vector<sigma::Ciphertext *>> storageQueue;
//std::mutex queueMutex;
//std::condition_variable cvRead;

void readData(size_t slots, float scale) {

    util::NPYReader reader(FILE_STORE_PATH + "gallery_x.npy", slots);

    while (true) {
        auto group = reader.read_data(scale);
        source_queue.push(group);
        if (group == nullptr) {
            break;
        }
    }
}

struct EncryptTask {
    cudaStream_t stream;
    sigma::Plaintext plaintext;
    sigma::Ciphertext ciphertext;
};

util::safe_queue<EncryptTask *> task_queue;

void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void *data) {
    auto task = reinterpret_cast<EncryptTask *>(data);
    auto ciphertext = new sigma::Ciphertext(task->ciphertext, false);
    store_queue.push(ciphertext);
    task_queue.push(task);
}

void processData(size_t poly_modulus_degree, double scale) {

    EncryptTask tasks[cuda_stream_num];
    for (auto &task: tasks) {
        cudaStreamCreate(&task.stream);
        task_queue.push(&task);
    }

    auto slots = poly_modulus_degree / 2;

    sigma::KernelProvider::initialize();
    sigma::EncryptionParameters params(sigma::scheme_type::ckks);
    params.set_poly_modulus_degree(poly_modulus_degree);
    auto modulus_bit_sizes = ConfigUtil.intVectorValueForKey("modulus_bit_sizes");
    params.set_coeff_modulus(sigma::CoeffModulus::Create(poly_modulus_degree, modulus_bit_sizes));
    sigma::SIGMAContext context(params);

    sigma::SecretKey secret_key;
    util::load_secret_key(context, secret_key, secret_key_data_path);

    secret_key.copy_to_device();

    sigma::CKKSEncoder encoder(context);
    sigma::Encryptor encryptor(context, secret_key);

    sigma::Ciphertext c1;
    c1.use_half_data() = true;
    encryptor.sample_symmetric_ckks_c1(c1);
    std::ofstream c1_ofs(encrypted_c1_data_path, std::ios::binary);
    c1.save(c1_ofs);
    c1_ofs.close();
    c1.copy_to_device();

    std::cout << "Encode and encrypt start" << std::endl;
    auto time_start = std::chrono::high_resolution_clock::now();

    while (true) {
        auto group = source_queue.pop();
        if (group == nullptr) {
            break;
        }
        while (true) {
            auto list = group->next_list();
            if (list == nullptr) {
                break;
            }

            auto task = task_queue.pop();
            auto &stream = task->stream;
            auto &plaintext = task->plaintext;
            auto &ciphertext = task->ciphertext;

            encoder.encode_float(list->get(), slots, scale, plaintext, stream);
            ciphertext.use_half_data() = true;
            encryptor.encrypt_symmetric_ckks(plaintext, ciphertext, c1, stream);
            ciphertext.retrieve_to_host(stream);

            cudaStreamAddCallback(stream, my_callback, (void *) (task), 0);
        }
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
    std::cout << "Encode and encrypt end [" << time_diff.count() << " milliseconds]" << std::endl;
    for (auto &task: tasks) {
        cudaStreamSynchronize(task.stream);
    }
    store_queue.push(nullptr);


}

// Function to simulate storing data to disk
void storeData() {

    std::ofstream ofs(encrypted_data_path, std::ios::binary);

    int count = 0;
    while (true) {
        auto ciphertext = store_queue.pop();
        if (ciphertext == nullptr && store_queue.empty()) {
            break;
        }

        ciphertext->save(ofs);

        delete ciphertext;
        count ++;
    }
    std::cout << "count = " << count << std::endl;
    ofs.close();
}

int main() {

//    std::cout << "Encode and encrypt start" << std::endl;
//    auto time_start = std::chrono::high_resolution_clock::now();

    size_t poly_modulus_degree = ConfigUtil.int64ValueForKey("poly_modulus_degree");
    auto slots = poly_modulus_degree / 2;
    size_t scale_power = ConfigUtil.int64ValueForKey("scale_power");
    double scale = pow(2.0, scale_power);

    std::thread readerThread(readData, std::ref(slots), std::ref(scale));
    std::thread processorThread(processData, std::ref(poly_modulus_degree), std::ref(scale));
    std::thread storeThread(storeData);

    if (readerThread.joinable()) readerThread.join();
    if (processorThread.joinable()) processorThread.join();
    if (storeThread.joinable()) storeThread.join();

//    size_t poly_modulus_degree = ConfigUtil.int64ValueForKey("poly_modulus_degree");
//    size_t scale_power = ConfigUtil.int64ValueForKey("scale_power");
//    double scale = pow(2.0, scale_power);
//    size_t customized_scale_power = ConfigUtil.int64ValueForKey("customized_scale_power");
//    double customized_scale = pow(2.0, customized_scale_power);
//
//
//    auto slots = poly_modulus_degree / 2;
//
//    size_t gallery_size = 0;
//    auto gallery_ptr = util::read_formatted_npy_data(FILE_STORE_PATH + "gallery_x.npy", slots, customized_scale,
//                                                     gallery_size);
//
//    sigma::KernelProvider::initialize();
//
//    sigma::EncryptionParameters params(sigma::scheme_type::ckks);
//    params.set_poly_modulus_degree(poly_modulus_degree);
//    auto modulus_bit_sizes = ConfigUtil.intVectorValueForKey("modulus_bit_sizes");
//    params.set_coeff_modulus(sigma::CoeffModulus::Create(poly_modulus_degree, modulus_bit_sizes));
////    params.setup_device_params(); // 初始化device相关参数
//    sigma::SIGMAContext context(params);
////    context.setup_device_params(); // 初始化device相关参数
//
////    sigma::PublicKey public_key;
//    sigma::SecretKey secret_key;
////    util::load_public_key(context, public_key, public_key_data_path);
//    util::load_secret_key(context, secret_key, secret_key_data_path);
//
//    secret_key.copy_to_device();
//
//    sigma::CKKSEncoder encoder(context);
//    sigma::Encryptor encryptor(context, secret_key);
//
//    sigma::Ciphertext c1;
//    c1.use_half_data() = true;
//    encryptor.sample_symmetric_ckks_c1(c1);
//    std::ofstream c1_ofs(encrypted_c1_data_path, std::ios::binary);
//    c1.save(c1_ofs);
//    c1_ofs.close();
//
//    c1.copy_to_device();
//
//    std::ofstream ofs(encrypted_data_path, std::ios::binary);
//
//    std::cout << "Encode and encrypt start" << std::endl;
//    auto time_start = std::chrono::high_resolution_clock::now();
//
//    sigma::Plaintext plain_vec;
//    sigma::Ciphertext ciphertext;
//    for (int i = 0; i < gallery_size; ++i) {
//        auto vec = gallery_ptr + (i * slots);
//
//        auto time_start0 = std::chrono::high_resolution_clock::now();
//
////        encoder.encode_double(vec, slots, scale, plain_vec);
//
////        auto time_end0 = std::chrono::high_resolution_clock::now();
////        auto time_diff0 = std::chrono::duration_cast<std::chrono::microseconds >(time_end0 - time_start0);
////        std::cout << "encrypt file end [" << time_diff0.count() << " microseconds]" << std::endl;
//
////        auto time_start1 = std::chrono::high_resolution_clock::now();
//
//        ciphertext.use_half_data() = true;
//        encryptor.encrypt_symmetric_ckks(plain_vec, ciphertext, c1);
//
//        ciphertext.retrieve_to_host();
//
////        auto time_end1 = std::chrono::high_resolution_clock::now();
////        auto time_diff1 = std::chrono::duration_cast<std::chrono::microseconds >(time_end1 - time_start1);
////        std::cout << "encrypt file end [" << time_diff1.count() << " microseconds]" << std::endl;
////        std::cout << std::endl << std::endl;
//
//        ciphertext.save(ofs);
////        std::cout << "encrypt end " << i << std::endl;  // TODO: remove @wangshuchao
//    }

//    auto time_end = std::chrono::high_resolution_clock::now();
//    auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
//    std::cout << "Encode and encrypt end [" << time_diff.count() << " milliseconds]" << std::endl;
//
//    c1.release_device_data();
//
//    ofs.close();
//
//    delete[] gallery_ptr;

    return 0;
}
