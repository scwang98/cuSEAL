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

SIGMA_C_FUNC EPQ_Create(void *copy, void **epq);

SIGMA_C_FUNC EPQ_Destroy(void *thisptr);

SIGMA_C_FUNC EPQ_ParametersSet(void *thisptr, bool *parameters_set);

SIGMA_C_FUNC EPQ_UsingFFT(void *thisptr, bool *using_fft);

SIGMA_C_FUNC EPQ_UsingNTT(void *thisptr, bool *using_ntt);

SIGMA_C_FUNC EPQ_UsingBatching(void *thisptr, bool *using_batching);

SIGMA_C_FUNC EPQ_UsingFastPlainLift(void *thisptr, bool *using_fast_plain_lift);

SIGMA_C_FUNC EPQ_UsingDescendingModulusChain(void *thisptr, bool *using_descending_modulus_chain);

SIGMA_C_FUNC EPQ_SecLevel(void *thisptr, int *sec_level);

SIGMA_C_FUNC EPQ_ParameterErrorName(void *thisptr, char *outstr, uint64_t *length);

SIGMA_C_FUNC EPQ_ParameterErrorMessage(void *thisptr, char *outstr, uint64_t *length);
