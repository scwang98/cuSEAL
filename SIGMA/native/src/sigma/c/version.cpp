// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// SIGMANet
#include "sigma/c/utilities.h"
#include "sigma/c/version.h"

// SIGMA
#include "sigma/util/config.h"

using namespace std;
using namespace sigma::c;

SIGMA_C_FUNC Version_Major(uint8_t *result)
{
    IfNullRet(result, E_POINTER);

    *result = (uint8_t)SIGMA_VERSION_MAJOR;
    return S_OK;
}

SIGMA_C_FUNC Version_Minor(uint8_t *result)
{
    IfNullRet(result, E_POINTER);

    *result = (uint8_t)SIGMA_VERSION_MINOR;
    return S_OK;
}

SIGMA_C_FUNC Version_Patch(uint8_t *result)
{
    IfNullRet(result, E_POINTER);

    *result = (uint8_t)SIGMA_VERSION_PATCH;
    return S_OK;
}
