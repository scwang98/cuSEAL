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

SIGMA_C_FUNC SecretKey_Create1(void **secret_key);

SIGMA_C_FUNC SecretKey_Create2(void *copy, void **secret_key);

SIGMA_C_FUNC SecretKey_Set(void *thisptr, void *assign);

SIGMA_C_FUNC SecretKey_Data(void *thisptr, void **data);

SIGMA_C_FUNC SecretKey_Destroy(void *thisptr);

SIGMA_C_FUNC SecretKey_ParmsId(void *thisptr, uint64_t *parms_id);

SIGMA_C_FUNC SecretKey_Pool(void *thisptr, void **pool);

SIGMA_C_FUNC SecretKey_SaveSize(void *thisptr, uint8_t compr_mode, int64_t *result);

SIGMA_C_FUNC SecretKey_Save(void *thisptr, uint8_t *outptr, uint64_t size, uint8_t compr_mode, int64_t *out_bytes);

SIGMA_C_FUNC SecretKey_UnsafeLoad(void *thisptr, void *context, uint8_t *inptr, uint64_t size, int64_t *in_bytes);

SIGMA_C_FUNC SecretKey_Load(void *thisptr, void *context, uint8_t *inptr, uint64_t size, int64_t *in_bytes);
