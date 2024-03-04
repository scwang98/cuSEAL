//
// Created by yuked on 2024/1/22.
//

#include <string>
#include <vector>

#ifndef CUSEAL_CLUSTERING_H
#define CUSEAL_CLUSTERING_H

namespace util {

    void clusteringThreadStep2(const std::vector<std::vector<double>> &xdata, const int k,
                               const std::vector<std::vector<double>> &centroids, std::vector<int> &localAssig,
                               int start, int end);

    std::pair<std::vector<int>, std::vector<std::vector<double>>>
    clusteringMultiThread(const std::vector<std::vector<double>> &xdata, const int k, const int maxIter);

} // util

#endif //CUSEAL_CLUSTERING_H


