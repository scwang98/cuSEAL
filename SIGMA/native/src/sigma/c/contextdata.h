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

SIGMA_C_FUNC ContextData_Destroy(void *thisptr);

SIGMA_C_FUNC ContextData_TotalCoeffModulus(void *thisptr, uint64_t *count, uint64_t *total_coeff_modulus);

SIGMA_C_FUNC ContextData_TotalCoeffModulusBitCount(void *thisptr, int *bit_count);

SIGMA_C_FUNC ContextData_Parms(void *thisptr, void **parms);

SIGMA_C_FUNC ContextData_Qualifiers(void *thisptr, void **epq);

SIGMA_C_FUNC ContextData_CoeffDivPlainModulus(void *thisptr, uint64_t *count, uint64_t *coeff_div);

SIGMA_C_FUNC ContextData_PlainUpperHalfThreshold(void *thisptr, uint64_t *puht);

SIGMA_C_FUNC ContextData_PlainUpperHalfIncrement(void *thisptr, uint64_t *count, uint64_t *puhi);

SIGMA_C_FUNC ContextData_UpperHalfThreshold(void *thisptr, uint64_t *count, uint64_t *uht);

SIGMA_C_FUNC ContextData_UpperHalfIncrement(void *thisptr, uint64_t *count, uint64_t *uhi);

SIGMA_C_FUNC ContextData_PrevContextData(void *thisptr, void **prev_data);

SIGMA_C_FUNC ContextData_NextContextData(void *thisptr, void **next_data);

SIGMA_C_FUNC ContextData_ChainIndex(void *thisptr, uint64_t *index);
