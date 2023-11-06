// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// STD
#include <unordered_map>

// SIGMANet
#include "sigma/c/sigmacontext.h"
#include "sigma/c/utilities.h"

// SIGMA
#include "sigma/context.h"
#include "sigma/util/locks.h"

using namespace std;
using namespace sigma;
using namespace sigma::util;
using namespace sigma::c;

SIGMA_C_FUNC SIGMAContext_Create(void *encryptionParams, bool expand_mod_chain, int sec_level, void **context)
{
    EncryptionParameters *encParams = FromVoid<EncryptionParameters>(encryptionParams);
    IfNullRet(encParams, E_POINTER);
    IfNullRet(context, E_POINTER);

    sec_level_type security_level = static_cast<sec_level_type>(sec_level);

    *context = new SIGMAContext(*encParams, expand_mod_chain, security_level);
    return S_OK;
}

SIGMA_C_FUNC SIGMAContext_Destroy(void *thisptr)
{
    SIGMAContext *context = FromVoid<SIGMAContext>(thisptr);
    IfNullRet(context, E_POINTER);

    delete context;
    return S_OK;
}

SIGMA_C_FUNC SIGMAContext_KeyParmsId(void *thisptr, uint64_t *parms_id)
{
    SIGMAContext *context = FromVoid<SIGMAContext>(thisptr);
    IfNullRet(context, E_POINTER);
    IfNullRet(parms_id, E_POINTER);

    CopyParmsId(context->key_parms_id(), parms_id);
    return S_OK;
}

SIGMA_C_FUNC SIGMAContext_FirstParmsId(void *thisptr, uint64_t *parms_id)
{
    SIGMAContext *context = FromVoid<SIGMAContext>(thisptr);
    IfNullRet(context, E_POINTER);
    IfNullRet(parms_id, E_POINTER);

    CopyParmsId(context->first_parms_id(), parms_id);
    return S_OK;
}

SIGMA_C_FUNC SIGMAContext_LastParmsId(void *thisptr, uint64_t *parms_id)
{
    SIGMAContext *context = FromVoid<SIGMAContext>(thisptr);
    IfNullRet(context, E_POINTER);
    IfNullRet(parms_id, E_POINTER);

    CopyParmsId(context->last_parms_id(), parms_id);
    return S_OK;
}

SIGMA_C_FUNC SIGMAContext_ParametersSet(void *thisptr, bool *params_set)
{
    SIGMAContext *context = FromVoid<SIGMAContext>(thisptr);
    IfNullRet(context, E_POINTER);
    IfNullRet(params_set, E_POINTER);

    *params_set = context->parameters_set();
    return S_OK;
}

SIGMA_C_FUNC SIGMAContext_KeyContextData(void *thisptr, void **context_data)
{
    SIGMAContext *context = FromVoid<SIGMAContext>(thisptr);
    IfNullRet(context, E_POINTER);
    IfNullRet(context_data, E_POINTER);

    // The pointer that is returned should not be deleted.
    auto data = context->key_context_data();
    *context_data = const_cast<SIGMAContext::ContextData *>(data.get());
    return S_OK;
}

SIGMA_C_FUNC SIGMAContext_FirstContextData(void *thisptr, void **context_data)
{
    SIGMAContext *context = FromVoid<SIGMAContext>(thisptr);
    IfNullRet(context, E_POINTER);
    IfNullRet(context_data, E_POINTER);

    // The pointer that is returned should not be deleted.
    auto data = context->first_context_data();
    *context_data = const_cast<SIGMAContext::ContextData *>(data.get());
    return S_OK;
}

SIGMA_C_FUNC SIGMAContext_LastContextData(void *thisptr, void **context_data)
{
    SIGMAContext *context = FromVoid<SIGMAContext>(thisptr);
    IfNullRet(context, E_POINTER);
    IfNullRet(context_data, E_POINTER);

    // The pointer that is returned should not be deleted.
    auto data = context->last_context_data();
    *context_data = const_cast<SIGMAContext::ContextData *>(data.get());
    return S_OK;
}

SIGMA_C_FUNC SIGMAContext_GetContextData(void *thisptr, uint64_t *parms_id, void **context_data)
{
    SIGMAContext *context = FromVoid<SIGMAContext>(thisptr);
    IfNullRet(context, E_POINTER);
    IfNullRet(parms_id, E_POINTER);
    IfNullRet(context_data, E_POINTER);

    // The pointer that is returned should not be deleted.
    parms_id_type parms;
    CopyParmsId(parms_id, parms);
    auto data = context->get_context_data(parms);
    *context_data = const_cast<SIGMAContext::ContextData *>(data.get());
    return S_OK;
}

SIGMA_C_FUNC SIGMAContext_UsingKeyswitching(void *thisptr, bool *using_keyswitching)
{
    SIGMAContext *context = FromVoid<SIGMAContext>(thisptr);
    IfNullRet(context, E_POINTER);
    IfNullRet(using_keyswitching, E_POINTER);

    *using_keyswitching = context->using_keyswitching();
    return S_OK;
}

SIGMA_C_FUNC SIGMAContext_ParameterErrorName(void *thisptr, char *outstr, uint64_t *length)
{
    SIGMAContext *context = FromVoid<SIGMAContext>(thisptr);
    IfNullRet(context, E_POINTER);
    IfNullRet(length, E_POINTER);

    const char *str = context->parameter_error_name();
    *length = static_cast<uint64_t>(strlen(str));

    if (nullptr != outstr)
    {
        memcpy(outstr, str, *length);
    }
    return S_OK;
}

SIGMA_C_FUNC SIGMAContext_ParameterErrorMessage(void *thisptr, char *outstr, uint64_t *length)
{
    SIGMAContext *context = FromVoid<SIGMAContext>(thisptr);
    IfNullRet(context, E_POINTER);
    IfNullRet(length, E_POINTER);

    const char *str = context->parameter_error_message();
    *length = static_cast<uint64_t>(strlen(str));

    if (nullptr != outstr)
    {
        memcpy(outstr, str, *length);
    }
    return S_OK;
}
