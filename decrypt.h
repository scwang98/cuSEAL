//
// Created by scwang on 2024/3/4.
//

#ifndef CUSEAL_DECRYPT_H
#define CUSEAL_DECRYPT_H

#include <string>

/**
 * 内积计算阶段

 @param secret_key_path 密钥存储路径，如"../data/secret_key.dat"
 @param results_path 比对结果存储路径，如"../data/ip_results/top_ip_results.json"
*/
int decrypt(const std::string &secret_key_path, const std::string &results_path);

#endif //CUSEAL_DECRYPT_H
