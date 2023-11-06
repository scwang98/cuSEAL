// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// SIGMANet
#include "sigma/c/serialization.h"
#include "sigma/c/utilities.h"

// SIGMA
#include "sigma/serialization.h"

using namespace std;
using namespace sigma;
using namespace sigma::c;

SIGMA_C_FUNC Serialization_SIGMAMagic(uint16_t *result)
{
    IfNullRet(result, E_POINTER);

    *result = Serialization::sigma_magic;
    return S_OK;
}

SIGMA_C_FUNC Serialization_SIGMAHeaderSize(uint8_t *result)
{
    IfNullRet(result, E_POINTER);

    *result = Serialization::sigma_header_size;
    return S_OK;
}

SIGMA_C_FUNC Serialization_IsSupportedComprMode(uint8_t compr_mode, bool *result)
{
    IfNullRet(result, E_POINTER);

    *result = Serialization::IsSupportedComprMode(compr_mode);
    return S_OK;
}

SIGMA_C_FUNC Serialization_ComprModeDefault(uint8_t *result)
{
    IfNullRet(result, E_POINTER);

    *result = static_cast<uint8_t>(Serialization::compr_mode_default);
    return S_OK;
}

SIGMA_C_FUNC Serialization_IsCompatibleVersion(uint8_t *headerptr, uint64_t size, bool *result)
{
    IfNullRet(headerptr, E_POINTER);
    IfNullRet(result, E_POINTER);
    if (size != static_cast<uint64_t>(sizeof(Serialization::SIGMAHeader)))
    {
        *result = false;
    }

    Serialization::SIGMAHeader header;
    memcpy(
        reinterpret_cast<sigma_byte *>(&header), reinterpret_cast<sigma_byte *>(headerptr),
        sizeof(Serialization::SIGMAHeader));
    *result = Serialization::IsCompatibleVersion(header);
    return S_OK;
}

SIGMA_C_FUNC Serialization_IsValidHeader(uint8_t *headerptr, uint64_t size, bool *result)
{
    IfNullRet(headerptr, E_POINTER);
    IfNullRet(result, E_POINTER);
    if (size != static_cast<uint64_t>(sizeof(Serialization::SIGMAHeader)))
    {
        *result = false;
    }

    Serialization::SIGMAHeader header;
    memcpy(
        reinterpret_cast<sigma_byte *>(&header), reinterpret_cast<sigma_byte *>(headerptr),
        sizeof(Serialization::SIGMAHeader));
    *result = Serialization::IsValidHeader(header);
    return S_OK;
}
