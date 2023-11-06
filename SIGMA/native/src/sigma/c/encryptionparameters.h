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

SIGMA_C_FUNC EncParams_Create1(uint8_t scheme, void **enc_params);

SIGMA_C_FUNC EncParams_Create2(void *copy, void **enc_params);

SIGMA_C_FUNC EncParams_Destroy(void *thisptr);

SIGMA_C_FUNC EncParams_Set(void *thisptr, void *assign);

SIGMA_C_FUNC EncParams_GetPolyModulusDegree(void *thisptr, uint64_t *degree);

SIGMA_C_FUNC EncParams_SetPolyModulusDegree(void *thisptr, uint64_t degree);

SIGMA_C_FUNC EncParams_GetCoeffModulus(void *thisptr, uint64_t *length, void **coeffs);

SIGMA_C_FUNC EncParams_SetCoeffModulus(void *thisptr, uint64_t length, void **coeffs);

SIGMA_C_FUNC EncParams_GetScheme(void *thisptr, uint8_t *scheme);

SIGMA_C_FUNC EncParams_GetParmsId(void *thisptr, uint64_t *parms_id);

SIGMA_C_FUNC EncParams_GetPlainModulus(void *thisptr, void **plain_modulus);

SIGMA_C_FUNC EncParams_SetPlainModulus1(void *thisptr, void *modulus);

SIGMA_C_FUNC EncParams_SetPlainModulus2(void *thisptr, uint64_t plain_modulus);

SIGMA_C_FUNC EncParams_Equals(void *thisptr, void *otherptr, bool *result);

SIGMA_C_FUNC EncParams_SaveSize(void *thisptr, uint8_t compr_mode, int64_t *result);

SIGMA_C_FUNC EncParams_Save(void *thisptr, uint8_t *outptr, uint64_t size, uint8_t compr_mode, int64_t *out_bytes);

SIGMA_C_FUNC EncParams_Load(void *thisptr, uint8_t *inptr, uint64_t size, int64_t *in_bytes);
