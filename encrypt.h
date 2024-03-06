//
// Created by scwang on 2024/3/4.
//

#ifndef CUSEAL_ENCRYPT_H
#define CUSEAL_ENCRYPT_H

#include <string>

/**
 * 底库编码加密阶段

 @param secret_key_path 密钥存储路径，如"../data/secret_key.dat"
 @param gallery_path 底库npy文件路径，如""../vectors/gallery_x.npy"
 @param encrypted_directory 加密后的底库存储路径(文件夹)，如"../data"
*/
void encrypt(const std::string &secret_key_path, const std::string &gallery_path, const std::string &encrypted_directory);

#endif //CUSEAL_ENCRYPT_H
