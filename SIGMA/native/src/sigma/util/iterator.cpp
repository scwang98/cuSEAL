// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "sigma/ciphertext.h"
#include "sigma/util/iterator.h"

namespace sigma
{
    namespace util
    {
        PolyIter::PolyIter(Ciphertext &ct) : self_type(ct.data(), ct.poly_modulus_degree(), ct.coeff_modulus_size())
        {}

        ConstPolyIter::ConstPolyIter(const Ciphertext &ct)
            : self_type(ct.data(), ct.poly_modulus_degree(), ct.coeff_modulus_size())
        {}

        ConstPolyIter::ConstPolyIter(Ciphertext &ct)
            : self_type(ct.data(), ct.poly_modulus_degree(), ct.coeff_modulus_size())
        {}
    } // namespace util
} // namespace sigma
