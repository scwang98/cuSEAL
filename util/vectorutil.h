//
// Created by scwang on 2023/11/12.
//

#include <string>
#include <vector>

#ifndef CUSEAL_VECTORUTIL_H
#define CUSEAL_VECTORUTIL_H

namespace util {

    std::vector<std::vector<double>> read_npy_data(const std::string &npy_name);

} // util

#endif //CUSEAL_VECTORUTIL_H
