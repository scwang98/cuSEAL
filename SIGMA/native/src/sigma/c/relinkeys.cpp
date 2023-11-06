// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// SIGMANet
#include "sigma/c/relinkeys.h"
#include "sigma/c/utilities.h"

// SIGMA
#include "sigma/relinkeys.h"

using namespace std;
using namespace sigma;
using namespace sigma::c;

SIGMA_C_FUNC RelinKeys_GetIndex(uint64_t key_power, uint64_t *index)
{
    IfNullRet(index, E_POINTER);

    try
    {
        *index = RelinKeys::get_index(key_power);
        return S_OK;
    }
    catch (const invalid_argument &)
    {
        return E_INVALIDARG;
    }
}
