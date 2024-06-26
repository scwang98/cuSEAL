// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "ciphertext.h"
#include "context.h"
#include "encryptionparams.h"
#include "publickey.h"
#include "randomgen.h"
#include "secretkey.h"
#include <cstdint>

namespace sigma
{
    namespace util
    {
        /**
        Generate a uniform ternary polynomial and store in RNS representation.

        @param[in] prng A uniform random generator
        @param[in] parms EncryptionParameters used to parameterize an RNS polynomial
        @param[out] destination Allocated space to store a random polynomial
        */
        void sample_poly_ternary(
            std::shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms,
            std::uint64_t *destination);

        /**
        Generate a polynomial from a normal distribution and store in RNS representation.

        @param[in] prng A uniform random generator
        @param[in] parms EncryptionParameters used to parameterize an RNS polynomial
        @param[out] destination Allocated space to store a random polynomial
        */
        void sample_poly_normal(
            std::shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms,
            std::uint64_t *destination);

        /**
        Generate a polynomial from a centered binomial distribution and store in RNS representation.

        @param[in] prng A uniform random generator.
        @param[in] parms EncryptionParameters used to parameterize an RNS polynomial
        @param[out] destination Allocated space to store a random polynomial
        */
        void sample_poly_cbd(
            std::shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms,
            std::uint64_t *destination);

        /**
        Generate a uniformly random polynomial and store in RNS representation.

        @param[in] prng A uniform random generator
        @param[in] parms EncryptionParameters used to parameterize an RNS polynomial
        @param[out] destination Allocated space to store a random polynomial
        */
        void sample_poly_uniform(
            std::shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms,
            std::uint64_t *destination);

        /**
        Generate a uniformly random polynomial and store in RNS representation.
        This implementation corresponds to Microsoft SIGMA 3.4 and earlier.

        @param[in] prng A uniform random generator
        @param[in] parms EncryptionParameters used to parameterize an RNS polynomial
        @param[out] destination Allocated space to store a random polynomial
        */
        void sample_poly_uniform_sigma_3_4(
            std::shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms,
            std::uint64_t *destination);

        /**
        Generate a uniformly random polynomial and store in RNS representation.
        This implementation corresponds to Microsoft SIGMA 3.5 and earlier.

        @param[in] prng A uniform random generator
        @param[in] parms EncryptionParameters used to parameterize an RNS polynomial
        @param[out] destination Allocated space to store a random polynomial
        */
        void sample_poly_uniform_sigma_3_5(
            std::shared_ptr<UniformRandomGenerator> prng, const EncryptionParameters &parms,
            std::uint64_t *destination);

        /**
        Create an encryption of zero with a public key and store in a ciphertext.

        @param[in] public_key The public key used for encryption
        @param[in] context The SIGMAContext containing a chain of ContextData
        @param[in] parms_id Indicates the level of encryption
        @param[in] is_ntt_form If true, store ciphertext in NTT form
        @param[out] destination The output ciphertext - an encryption of zero
        */
        void encrypt_zero_asymmetric(
            const PublicKey &public_key, const SIGMAContext &context, parms_id_type parms_id, bool is_ntt_form,
            Ciphertext &destination);

        /**
        Create an encryption of zero with a secret key and store in a ciphertext.

        @param[out] destination The output ciphertext - an encryption of zero
        @param[in] secret_key The secret key used for encryption
        @param[in] context The SIGMAContext containing a chain of ContextData
        @param[in] parms_id Indicates the level of encryption
        @param[in] is_ntt_form If true, store ciphertext in NTT form
        @param[in] save_seed If true, the second component of ciphertext is
        replaced with the random seed used to sample this component
        */
        void encrypt_zero_symmetric(
            const SecretKey &secret_key, const SIGMAContext &context, parms_id_type parms_id, bool is_ntt_form,
            bool save_seed, Ciphertext &destination);
    } // namespace util
} // namespace sigma
