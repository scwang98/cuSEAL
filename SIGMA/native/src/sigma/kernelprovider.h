#pragma once

#include <stdexcept>

namespace sigma {

    class KernelProvider {

        static bool initialized;

    public:

        static void checkInitialized();

        static void initialize();

        template<typename T>
        static T *malloc(size_t length);

        template<typename T>
        static void free(T *pointer);

        template<typename T>
        static void copy(T *deviceDestPtr, const T *hostFromPtr, size_t length);

        template<typename T>
        static void copyOnDevice(T *deviceDestPtr, const T *deviceFromPtr, size_t length);

        template<typename T>
        static void retrieve(T *hostDestPtr, const T *deviceFromPtr, size_t length);

        template<typename T>
        static void memsetZero(T *devicePtr, size_t length);

    };

}
