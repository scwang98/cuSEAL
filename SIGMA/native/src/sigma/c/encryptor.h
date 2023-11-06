// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

///////////////////////////////////////////////////////////////////////////
//
// This API is provided as a simple interface for Microsoft SIGMA library
// that can be PInvoked by .Net code.
//
///////////////////////////////////////////////////////////////////////////

#include "sigma/c/defines.h"
#include <stdint.h>

SIGMA_C_FUNC Encryptor_Create(void *context, void *public_key, void *secret_key, void **encryptor);

SIGMA_C_FUNC Encryptor_SetPublicKey(void *thisptr, void *public_key);

SIGMA_C_FUNC Encryptor_SetSecretKey(void *thisptr, void *secret_key);

SIGMA_C_FUNC Encryptor_Encrypt(void *thisptr, void *plaintext, void *destination, void *pool_handle);

SIGMA_C_FUNC Encryptor_EncryptZero1(void *thisptr, uint64_t *parms_id, void *destination, void *pool_handle);

SIGMA_C_FUNC Encryptor_EncryptZero2(void *thisptr, void *destination, void *pool_handle);

SIGMA_C_FUNC Encryptor_EncryptSymmetric(
    void *thisptr, void *plaintext, bool save_seed, void *destination, void *pool_handle);

SIGMA_C_FUNC Encryptor_EncryptZeroSymmetric1(
    void *thisptr, uint64_t *parms_id, bool save_seed, void *destination, void *pool_handle);

SIGMA_C_FUNC Encryptor_EncryptZeroSymmetric2(void *thisptr, bool save_seed, void *destination, void *pool_handle);

SIGMA_C_FUNC Encryptor_Destroy(void *thisptr);
