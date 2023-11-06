// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// SIGMANet
#include "sigma/c/galoiskeys.h"
#include "sigma/c/utilities.h"

// SIGMA
#include "sigma/galoiskeys.h"

using namespace std;
using namespace sigma;
using namespace sigma::c;

SIGMA_C_FUNC GaloisKeys_GetIndex(uint32_t galois_elt, uint64_t *index)
{
    IfNullRet(index, E_POINTER);

    try
    {
        *index = GaloisKeys::get_index(galois_elt);
        return S_OK;
    }
    catch (const invalid_argument &)
    {
        return E_INVALIDARG;
    }
}
