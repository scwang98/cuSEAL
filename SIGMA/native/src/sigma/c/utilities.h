// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// STD
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// SIGMANet
#include "sigma/c/defines.h"

// SIGMA
#include "sigma/encryptionparams.h"

namespace sigma
{
    class Modulus;
    class SIGMAContext;
    class MemoryPoolHandle;
} // namespace sigma

namespace sigma
{
    namespace c
    {
        /**
        Return a pointer of the given type from a void pointer.
        */
        template <class T>
        inline T *FromVoid(void *voidptr)
        {
            T *result = reinterpret_cast<T *>(voidptr);
            return result;
        }

        /**
        Get MemoryPoolHandle from a void pointer.
        Returns a default if void pointer is null.
        */
        std::unique_ptr<sigma::MemoryPoolHandle> MemHandleFromVoid(void *voidptr);

        /**
        Build and array of Modulus pointers from a vector
        */
        void BuildModulusPointers(const std::vector<sigma::Modulus> &in_mods, uint64_t *length, void **out_mods);

        /**
        Get a parms_id_type from an uint64_t pointer
        */
        inline void CopyParmsId(const uint64_t *src, sigma::parms_id_type &dest)
        {
            if (nullptr != src)
            {
                std::copy_n(src, dest.size(), std::begin(dest));
            }
        }

        /**
        Copy parms_id_type to a uint64_t pointer
        */
        inline void CopyParmsId(const sigma::parms_id_type &src, uint64_t *dest)
        {
            if (nullptr != dest)
            {
                std::copy_n(std::cbegin(src), src.size(), dest);
            }
        }

        /**
        Convert std::string to char* with null terminator
        */
        HRESULT ToStringHelper(const std::string &str, char *outstr, uint64_t *length);

        /**
        Convert const char * to char* with null terminator
        */
        HRESULT ToStringHelper2(const char *str, char *outstr, uint64_t *length);
    } // namespace c
} // namespace sigma
