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

SIGMA_C_FUNC Evaluator_Create(void *context, void **evaluator);

SIGMA_C_FUNC Evaluator_Destroy(void *thisptr);

SIGMA_C_FUNC Evaluator_Negate(void *thisptr, void *encrypted, void *destination);

SIGMA_C_FUNC Evaluator_Add(void *thisptr, void *encrypted1, void *encrypted2, void *destination);

SIGMA_C_FUNC Evaluator_AddMany(void *thisptr, uint64_t count, void **encrypteds, void *destination);

SIGMA_C_FUNC Evaluator_AddPlain(void *thisptr, void *encrypted, void *plain, void *destination);

SIGMA_C_FUNC Evaluator_Sub(void *thisptr, void *encrypted1, void *encrypted2, void *destination);

SIGMA_C_FUNC Evaluator_SubPlain(void *thisptr, void *encrypted, void *plain, void *destination);

SIGMA_C_FUNC Evaluator_Multiply(void *thisptr, void *encrypted1, void *encrypted2, void *destination, void *pool);

SIGMA_C_FUNC Evaluator_MultiplyMany(
    void *thisptr, uint64_t count, void **encrypteds, void *relin_keys, void *destination, void *pool);

SIGMA_C_FUNC Evaluator_MultiplyPlain(void *thisptr, void *encrypted, void *plain, void *destination, void *pool);

SIGMA_C_FUNC Evaluator_Square(void *thisptr, void *encrypted, void *destination, void *pool);

SIGMA_C_FUNC Evaluator_Relinearize(void *thisptr, void *encrypted, void *relinKeys, void *destination, void *pool);

SIGMA_C_FUNC Evaluator_ModSwitchToNext1(void *thisptr, void *encrypted, void *destination, void *pool);

SIGMA_C_FUNC Evaluator_ModSwitchTo1(void *thisptr, void *encrypted, uint64_t *parms_id, void *destination, void *pool);

SIGMA_C_FUNC Evaluator_ModSwitchToNext2(void *thisptr, void *plain, void *destination);

SIGMA_C_FUNC Evaluator_ModSwitchTo2(void *thisptr, void *plain, uint64_t *parms_id, void *destination);

SIGMA_C_FUNC Evaluator_RescaleToNext(void *thisptr, void *encrypted, void *destination, void *pool);

SIGMA_C_FUNC Evaluator_RescaleTo(void *thisptr, void *encrypted, uint64_t *parms_id, void *destination, void *pool);

SIGMA_C_FUNC Evaluator_ModReduceToNext(void *thisptr, void *encrypted, void *destination, void *pool);

SIGMA_C_FUNC Evaluator_ModReduceTo(void *thisptr, void *encrypted, uint64_t *parms_id, void *destination, void *pool);

SIGMA_C_FUNC Evaluator_Exponentiate(
    void *thisptr, void *encrypted, uint64_t exponent, void *relin_keys, void *destination, void *pool);

SIGMA_C_FUNC Evaluator_TransformToNTT1(
    void *thisptr, void *plain, uint64_t *parms_id, void *destination_ntt, void *pool);

SIGMA_C_FUNC Evaluator_TransformToNTT2(void *thisptr, void *encrypted, void *destination_ntt);

SIGMA_C_FUNC Evaluator_TransformFromNTT(void *thisptr, void *encrypted_ntt, void *destination);

SIGMA_C_FUNC Evaluator_ApplyGalois(
    void *thisptr, void *encrypted, uint32_t galois_elt, void *galois_keys, void *destination, void *pool);

SIGMA_C_FUNC Evaluator_RotateRows(
    void *thisptr, void *encrypted, int steps, void *galoisKeys, void *destination, void *pool);

SIGMA_C_FUNC Evaluator_RotateColumns(void *thisptr, void *encrypted, void *galois_keys, void *destination, void *pool);

SIGMA_C_FUNC Evaluator_RotateVector(
    void *thisptr, void *encrypted, int steps, void *galois_keys, void *destination, void *pool);

SIGMA_C_FUNC Evaluator_ComplexConjugate(
    void *thisptr, void *encrypted, void *galois_keys, void *destination, void *pool);

SIGMA_C_FUNC Evaluator_ContextUsingKeyswitching(void *thisptr, bool *using_keyswitching);
