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

SIGMA_C_FUNC MemoryPoolHandle_Create1(void **handle);

SIGMA_C_FUNC MemoryPoolHandle_Create2(void *otherptr, void **handle);

SIGMA_C_FUNC MemoryPoolHandle_Destroy(void *thisptr);

SIGMA_C_FUNC MemoryPoolHandle_Set(void *thisptr, void *assignptr);

SIGMA_C_FUNC MemoryPoolHandle_Global(void **handle);

SIGMA_C_FUNC MemoryPoolHandle_ThreadLocal(void **handle);

SIGMA_C_FUNC MemoryPoolHandle_New(bool clear_on_destruction, void **handle);

SIGMA_C_FUNC MemoryPoolHandle_PoolCount(void *thisptr, uint64_t *count);

SIGMA_C_FUNC MemoryPoolHandle_AllocByteCount(void *thisptr, uint64_t *count);

SIGMA_C_FUNC MemoryPoolHandle_UseCount(void *thisptr, long *count);

SIGMA_C_FUNC MemoryPoolHandle_IsInitialized(void *thisptr, bool *result);

SIGMA_C_FUNC MemoryPoolHandle_Equals(void *thisptr, void *otherptr, bool *result);
