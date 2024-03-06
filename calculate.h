//
// Created by scwang on 2024/3/4.
//

#ifndef CUSEAL_CALCULATE_H
#define CUSEAL_CALCULATE_H

#include <string>

/**
 * 内积计算阶段

 @param gallery_path 探测库npy文件路径，如""../vectors/probe_x.npy"
 @param encrypted_directory 加密后的底库存储路径(文件夹)，如"../data"
 @param result_directory 计算结果存储路径(文件夹)，如"../data/ip_results"
*/
void calculate(const std::string &probe_path, const std::string &encrypted_directory, const std::string &result_directory);

#endif //CUSEAL_CALCULATE_H
