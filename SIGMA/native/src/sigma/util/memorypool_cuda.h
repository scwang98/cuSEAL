#pragma once

#include "../kernelprovider.h"
#include <map>


namespace sigma::util {

    class MemoryPoolCuda {

        const size_t preservedMemory = 1024 * 1024 * 32;

    private:
        static MemoryPoolCuda singleton;

        MemoryPoolCuda();

        std::multimap<size_t, void *> pointers;
        size_t allocated;
        size_t totalMemory;

        size_t release() {
            if (pointers.size() == 0) return 0;
            size_t released = 0;
            for (auto &pair: pointers) {
                KernelProvider::free(pair.second);
                released += pair.first;
            }
            printf("[MemoryPoolCuda] Released %ld bytes\n", released);
            allocated -= released;
            pointers.clear();
            return released;
        }

        void *tryAllocate(size_t require);

        inline void *get(size_t require) {
            auto iterator = pointers.lower_bound(require);
            if ((iterator == pointers.end()) || (iterator->first > require * 2)) {
                return tryAllocate(require);
            } else {
                void *p = iterator->second;
                pointers.erase(iterator);
                return p;
            }
        }

        inline void insert(void *ptr, size_t size) {
            pointers.insert(std::make_pair(size, ptr));
        }

        ~MemoryPoolCuda() {
            release();
        }

    public:

        inline static void *Get(size_t require) {
            return singleton.get(require);
        }

        inline static void Free(void *ptr, size_t size) {
            singleton.insert(ptr, size);
        }

    };

}

