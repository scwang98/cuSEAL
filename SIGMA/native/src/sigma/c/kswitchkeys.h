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

SIGMA_C_FUNC KSwitchKeys_Create1(void **kswitch_keys);

SIGMA_C_FUNC KSwitchKeys_Create2(void *copy, void **kswitch_keys);

SIGMA_C_FUNC KSwitchKeys_Destroy(void *thisptr);

SIGMA_C_FUNC KSwitchKeys_Set(void *thisptr, void *assign);

SIGMA_C_FUNC KSwitchKeys_Size(void *thisptr, uint64_t *size);

SIGMA_C_FUNC KSwitchKeys_RawSize(void *thisptr, uint64_t *key_count);

SIGMA_C_FUNC KSwitchKeys_GetKeyList(void *thisptr, uint64_t index, uint64_t *count, void **key_list);

SIGMA_C_FUNC KSwitchKeys_ClearDataAndReserve(void *thisptr, uint64_t size);

SIGMA_C_FUNC KSwitchKeys_AddKeyList(void *thisptr, uint64_t count, void **key_list);

SIGMA_C_FUNC KSwitchKeys_GetParmsId(void *thisptr, uint64_t *parms_id);

SIGMA_C_FUNC KSwitchKeys_SetParmsId(void *thisptr, uint64_t *parms_id);

SIGMA_C_FUNC KSwitchKeys_Pool(void *thisptr, void **pool);

SIGMA_C_FUNC KSwitchKeys_SaveSize(void *thisptr, uint8_t compr_mode, int64_t *result);

SIGMA_C_FUNC KSwitchKeys_Save(void *thisptr, uint8_t *outptr, uint64_t size, uint8_t compr_mode, int64_t *out_bytes);

SIGMA_C_FUNC KSwitchKeys_UnsafeLoad(void *thisptr, void *context, uint8_t *inptr, uint64_t size, int64_t *in_bytes);

SIGMA_C_FUNC KSwitchKeys_Load(void *thisptr, void *context, uint8_t *inptr, uint64_t size, int64_t *in_bytes);
