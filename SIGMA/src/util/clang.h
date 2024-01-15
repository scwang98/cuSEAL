// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#if SIGMA_COMPILER == SIGMA_COMPILER_CLANG

// We require clang >= 5
#if (__clang_major__ < 5) || not defined(__cplusplus)
#error "SIGMA requires __clang_major__  >= 5"
#endif

// Read in config.h
#include "util/config.h"

#ifdef SIGMA_USE_ALIGNED_ALLOC
#include <cstdlib>
#define SIGMA_MALLOC(size) \
    static_cast<sigma_byte *>((((size)&63) == 0) ? ::aligned_alloc(64, (size)) : std::malloc((size)))
#define SIGMA_FREE(ptr) std::free(ptr)
#endif

// Are intrinsics enabled?
#ifdef SIGMA_USE_INTRIN
#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(EMSCRIPTEN)
#include <wasm_simd128.h>
#else
#include <x86intrin.h>
#endif

#ifdef SIGMA_USE___BUILTIN_CLZLL
#define SIGMA_MSB_INDEX_UINT64(result, value)                                 \
    {                                                                        \
        *result = 63UL - static_cast<unsigned long>(__builtin_clzll(value)); \
    }
#endif

#ifdef SIGMA_USE___INT128
__extension__ typedef __int128 int128_t;
__extension__ typedef unsigned __int128 uint128_t;
#define SIGMA_MULTIPLY_UINT64_HW64(operand1, operand2, hw64)                                 \
    do                                                                                      \
    {                                                                                       \
        *hw64 = static_cast<unsigned long long>(                                            \
            ((static_cast<uint128_t>(operand1) * static_cast<uint128_t>(operand2)) >> 64)); \
    } while (false)

#define SIGMA_MULTIPLY_UINT64(operand1, operand2, result128)              \
    do                                                                   \
    {                                                                    \
        uint128_t product = static_cast<uint128_t>(operand1) * operand2; \
        result128[0] = static_cast<unsigned long long>(product);         \
        result128[1] = static_cast<unsigned long long>(product >> 64);   \
    } while (false)

#define SIGMA_DIVIDE_UINT128_UINT64(numerator, denominator, result)                                 \
    do                                                                                             \
    {                                                                                              \
        uint128_t n, q;                                                                            \
        n = (static_cast<uint128_t>(numerator[1]) << 64) | (static_cast<uint128_t>(numerator[0])); \
        q = n / denominator;                                                                       \
        n -= q * denominator;                                                                      \
        numerator[0] = static_cast<std::uint64_t>(n);                                              \
        numerator[1] = 0;                                                                          \
        result[0] = static_cast<std::uint64_t>(q);                                                 \
        result[1] = static_cast<std::uint64_t>(q >> 64);                                           \
    } while (false)
#endif

#ifdef SIGMA_USE__ADDCARRY_U64
#define SIGMA_ADD_CARRY_UINT64(operand1, operand2, carry, result) _addcarry_u64(carry, operand1, operand2, result)
#endif

#ifdef SIGMA_USE__SUBBORROW_U64
#define SIGMA_SUB_BORROW_UINT64(operand1, operand2, borrow, result) _subborrow_u64(borrow, operand1, operand2, result)
#endif

#endif // SIGMA_USE_INTRIN

#endif
