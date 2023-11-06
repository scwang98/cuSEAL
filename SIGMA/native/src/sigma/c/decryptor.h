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

SIGMA_C_FUNC Decryptor_Create(void *context, void *secret_key, void **decryptor);

SIGMA_C_FUNC Decryptor_Destroy(void *thisptr);

SIGMA_C_FUNC Decryptor_Decrypt(void *thisptr, void *encrypted, void *destination);

SIGMA_C_FUNC Decryptor_InvariantNoiseBudget(void *thisptr, void *encrypted, int *invariant_noise_budget);
