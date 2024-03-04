//
// Created by yuked on 2024/1/22.
//

#include <iostream>
#include "clustering.h"
#include <cmath>
#include <thread>
#include <algorithm>

namespace util {

    void clusteringThreadStep2(const std::vector<std::vector<double>> &xdata, const int k,
                               const std::vector<std::vector<double>> &centroids, std::vector<int> &localAssig,
                               int start, int end) {
        std::vector<int> temp;
        for (int i = start; i < end; ++i) {
            double maxSimi = -std::numeric_limits<double>::max();
            int maxCentroidIndex = -1;

            for (int j = 0; j < k; ++j) {
                double simi = 0.0;
                for (int d = 0; d < xdata[i].size(); ++d)
                    simi += xdata[i][d] * centroids[j][d];

                if (simi > maxSimi) {
                    maxSimi = simi;
                    maxCentroidIndex = j;
                }
            }
            temp.push_back(maxCentroidIndex);       //用中间值直接整体赋值才行
            //localAssig.push_back(maxCentroidIndex);  //pushback只会在所有初始值-1后面，直接赋值元素会报错
        }
        localAssig = temp;
    }

    std::pair<std::vector<int>, std::vector<std::vector<double>>>
    clusteringMultiThread(const std::vector<std::vector<double>> &xdata, const int k, const int maxIter) {
        const int dataSize = xdata.size();
        const int vectorSize = xdata[0].size();

        std::vector<int> assignments(dataSize, -1);
        std::vector<std::vector<double>> centroids(k, std::vector<double>(vectorSize));

        // step1：随机选择K个初始聚类中心
        for (int i = 0; i < k; ++i)
            centroids[i] = xdata[i];

        bool centroidsChanged = true;
        int iter = 0;

        while (centroidsChanged && iter < maxIter) {
            centroidsChanged = false;

            // step2：多线程计算data与聚类中心的内积
            const int numThreads = 16;
            std::vector<std::thread> threads;       //创建一个存储线程对象的向量
            std::vector<std::vector<int>> localAssignments(numThreads,
                                                           std::vector<int>(dataSize / numThreads)); //存储每个线程的局部计算结果
            for (int i = 0; i < numThreads; ++i) {          //遍历每个线程的索引
                int start = i * (dataSize / numThreads);
                int end = (i == numThreads - 1) ? dataSize : (i + 1) * (dataSize / numThreads);
                threads.push_back(std::thread(clusteringThreadStep2, std::cref(xdata), k, std::cref(centroids),
                                              std::ref(localAssignments[i]), start,
                                              end));          //创建一个新的线程，并将其加入到 threads 向量中。每个线程都会调用 clusteringThreadStep2 函数，同时传入对应的参数，包括输入数据 xdata、聚类数 k、聚类中心 centroids、局部分配结果 localAssignments[i]，以及数据范围的起始和结束索引
            }

            for (auto &t: threads)
                if (t.joinable())   // 防止重复调用 join 导致异常
                    t.join();

            // step3：合并每个线程的计算结果
            for (int i = 0; i < numThreads; ++i)
                for (int j = 0; j < (dataSize / numThreads); ++j)
                    if (assignments[i * (dataSize / numThreads) + j] != localAssignments[i][j])
                        assignments[i * (dataSize / numThreads) + j] = localAssignments[i][j];

            // step4：更新聚类中心
            std::vector<std::vector<double>> newCentroids(k, std::vector<double>(vectorSize, 0.0));
            std::vector<int> clusterCounts(k, 0);

            for (int i = 0; i < dataSize; ++i) {
                int clusterIndex = assignments[i];
                for (size_t j = 0; j < vectorSize; ++j)
                    newCentroids[clusterIndex][j] += xdata[i][j];          //每个簇累加
            }

            for (int i = 0; i < k; ++i) {
                clusterCounts[i] = std::count(assignments.begin(), assignments.end(), i);   //每个簇中元素个数
                if (clusterCounts[i] > 0) {
                    for (size_t j = 0; j < newCentroids[i].size(); ++j)    //更新
                        newCentroids[i][j] /= clusterCounts[i];
                    if (centroids[i] != newCentroids[i]) {       // 检查聚类中心是否发生变化
                        centroids[i] = newCentroids[i];
                        centroidsChanged = true;
                    }
                }
            }

            iter++;
        }//while
        return std::make_pair(assignments, centroids);
    }//clusteringMultiThread

}//util