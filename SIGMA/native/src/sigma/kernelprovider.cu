
#include "kernelprovider.h"

namespace sigma {

    bool KernelProvider::initialized = false;

    void KernelProvider::checkInitialized() {
        if (!initialized)
            throw std::invalid_argument("KernelProvider not initialized.");
    }

    void KernelProvider::initialize() {
        cudaSetDevice(0);
        initialized = true;
    }

    template<typename T>
    T *KernelProvider::malloc(size_t length) {
        checkInitialized();
        if (length == 0) return nullptr;
        T *ret;
        auto status = cudaMalloc((void **) &ret, length * sizeof(T));
        if (status != cudaSuccess)
            throw std::runtime_error("Cuda Malloc failed.");
        return ret;
    }

    template<typename T>
    void KernelProvider::free(T *pointer) {
        checkInitialized();
        cudaFree(pointer);
    }

    template<typename T>
    void KernelProvider::copy(T *deviceDestPtr, const T *hostFromPtr, size_t length) {
        checkInitialized();
        if (length == 0) return;
        auto status = cudaMemcpy(deviceDestPtr, hostFromPtr, length * sizeof(T), cudaMemcpyHostToDevice);
        if (status != cudaSuccess)
            throw std::runtime_error("Cuda copy from host to device failed.");
    }

    template<typename T>
    void KernelProvider::copyOnDevice(T *deviceDestPtr, const T *deviceFromPtr, size_t length) {
        checkInitialized();
        if (length == 0) return;
        auto status = cudaMemcpy(deviceDestPtr, deviceFromPtr, length * sizeof(T), cudaMemcpyDeviceToDevice);
        if (status != cudaSuccess)
            throw std::runtime_error("Cuda copy on device failed.");
    }

    template<typename T>
    void KernelProvider::retrieve(T *hostDestPtr, const T *deviceFromPtr, size_t length) {
        checkInitialized();
        if (length == 0) return;
        auto status = cudaMemcpy(hostDestPtr, deviceFromPtr, length * sizeof(T), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess)
            throw std::runtime_error("Cuda retrieve from device to host failed.");
    }

    template<typename T>
    void KernelProvider::memsetZero(T *devicePtr, size_t length) {
        if (length == 0) return;
        cudaMemset(devicePtr, 0, sizeof(T) * length);
    }
}