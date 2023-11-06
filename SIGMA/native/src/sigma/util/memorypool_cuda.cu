
#include "memorypool_cuda.h"
#include "cuda_runtime.h"



namespace sigma::util {

    MemoryPoolCuda MemoryPoolCuda::singleton;

    MemoryPoolCuda::MemoryPoolCuda() {
        allocated = 0;
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, 0);
        totalMemory = props.totalGlobalMem;
        printf("[MemoryPoolCuda] Total Memory = %ld bytes\n", totalMemory);
    }

    void *MemoryPoolCuda::tryAllocate(size_t require) {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        if (free < require + preservedMemory) release();
        allocated += require;
        return reinterpret_cast<void *>(KernelProvider::malloc<char>(require));
    }

}

