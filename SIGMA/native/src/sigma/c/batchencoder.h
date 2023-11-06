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

SIGMA_C_FUNC BatchEncoder_Create(void *context, void **batch_encoder);

SIGMA_C_FUNC BatchEncoder_Destroy(void *thisptr);

SIGMA_C_FUNC BatchEncoder_Encode1(void *thisptr, uint64_t count, uint64_t *values, void *destination);

SIGMA_C_FUNC BatchEncoder_Encode2(void *thisptr, uint64_t count, int64_t *values, void *destination);

SIGMA_C_FUNC BatchEncoder_Decode1(void *thisptr, void *plain, uint64_t *count, uint64_t *destination, void *pool);

SIGMA_C_FUNC BatchEncoder_Decode2(void *thisptr, void *plain, uint64_t *count, int64_t *destination, void *pool);

SIGMA_C_FUNC BatchEncoder_GetSlotCount(void *thisptr, uint64_t *slot_count);
