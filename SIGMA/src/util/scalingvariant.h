// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "context.h"
#include "memorymanager.h"
#include "plaintext.cuh"
#include "util/iterator.h"
#include <cstdint>

namespace sigma
{
    namespace util
    {
        void add_plain_without_scaling_variant(
            const Plaintext &plain, const SIGMAContext::ContextData &context_data, RNSIter destination);

        void sub_plain_without_scaling_variant(
            const Plaintext &plain, const SIGMAContext::ContextData &context_data, RNSIter destination);

        void multiply_add_plain_with_scaling_variant(
            const Plaintext &plain, const SIGMAContext::ContextData &context_data, RNSIter destination);

        void multiply_sub_plain_with_scaling_variant(
            const Plaintext &plain, const SIGMAContext::ContextData &context_data, RNSIter destination);
    } // namespace util
} // namespace sigma
