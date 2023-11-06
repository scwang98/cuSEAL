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

SIGMA_C_FUNC SIGMAContext_Create(void *encryptionParams, bool expand_mod_chain, int sec_level, void **context);

SIGMA_C_FUNC SIGMAContext_Destroy(void *thisptr);

SIGMA_C_FUNC SIGMAContext_KeyParmsId(void *thisptr, uint64_t *parms_id);

SIGMA_C_FUNC SIGMAContext_FirstParmsId(void *thisptr, uint64_t *parms_id);

SIGMA_C_FUNC SIGMAContext_LastParmsId(void *thisptr, uint64_t *parms_id);

SIGMA_C_FUNC SIGMAContext_ParametersSet(void *thisptr, bool *params_set);

SIGMA_C_FUNC SIGMAContext_KeyContextData(void *thisptr, void **context_data);

SIGMA_C_FUNC SIGMAContext_FirstContextData(void *thisptr, void **context_data);

SIGMA_C_FUNC SIGMAContext_LastContextData(void *thisptr, void **context_data);

SIGMA_C_FUNC SIGMAContext_GetContextData(void *thisptr, uint64_t *parms_id, void **context_data);

SIGMA_C_FUNC SIGMAContext_UsingKeyswitching(void *thisptr, bool *using_keyswitching);

SIGMA_C_FUNC SIGMAContext_ParameterErrorName(void *thisptr, char *outstr, uint64_t *length);

SIGMA_C_FUNC SIGMAContext_ParameterErrorMessage(void *thisptr, char *outstr, uint64_t *length);
