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

SIGMA_C_FUNC MemoryManager_GetPool1(int prof_opt, bool clear_on_destruction, void **pool_handle);

SIGMA_C_FUNC MemoryManager_GetPool2(void **pool_handle);

SIGMA_C_FUNC MemoryManager_SwitchProfile(void *new_profile);

SIGMA_C_FUNC MMProf_CreateGlobal(void **profile);

SIGMA_C_FUNC MMProf_CreateFixed(void *pool, void **profile);

SIGMA_C_FUNC MMProf_CreateNew(void **profile);

SIGMA_C_FUNC MMProf_CreateThreadLocal(void **profile);

SIGMA_C_FUNC MMProf_GetPool(void *thisptr, void **pool_handle);

SIGMA_C_FUNC MMProf_Destroy(void *thisptr);
