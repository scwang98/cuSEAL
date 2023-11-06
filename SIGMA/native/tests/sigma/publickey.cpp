// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "sigma/context.h"
#include "sigma/keygenerator.h"
#include "sigma/modulus.h"
#include "sigma/publickey.h"
#include "gtest/gtest.h"

using namespace sigma;
using namespace std;

namespace sigmatest
{
    TEST(PublicKeyTest, SaveLoadPublicKey)
    {
        auto save_load_public_key = [](scheme_type scheme) {
            stringstream stream;
            {
                EncryptionParameters parms(scheme);
                parms.set_poly_modulus_degree(64);
                parms.set_plain_modulus(1 << 6);
                parms.set_coeff_modulus(CoeffModulus::Create(64, { 60 }));

                SIGMAContext context(parms, false, sec_level_type::none);
                KeyGenerator keygen(context);

                PublicKey pk;
                keygen.create_public_key(pk);
                ASSERT_TRUE(pk.parms_id() == context.key_parms_id());
                pk.save(stream);

                PublicKey pk2;
                pk2.load(context, stream);

                ASSERT_EQ(pk.data().dyn_array().size(), pk2.data().dyn_array().size());
                for (size_t i = 0; i < pk.data().dyn_array().size(); i++)
                {
                    ASSERT_EQ(pk.data().data()[i], pk2.data().data()[i]);
                }
                ASSERT_TRUE(pk.parms_id() == pk2.parms_id());
            }
            {
                EncryptionParameters parms(scheme);
                parms.set_poly_modulus_degree(256);
                parms.set_plain_modulus(1 << 20);
                parms.set_coeff_modulus(CoeffModulus::Create(256, { 30, 40 }));

                SIGMAContext context(parms, false, sec_level_type::none);
                KeyGenerator keygen(context);

                PublicKey pk;
                keygen.create_public_key(pk);
                ASSERT_TRUE(pk.parms_id() == context.key_parms_id());
                pk.save(stream);

                PublicKey pk2;
                pk2.load(context, stream);

                ASSERT_EQ(pk.data().dyn_array().size(), pk2.data().dyn_array().size());
                for (size_t i = 0; i < pk.data().dyn_array().size(); i++)
                {
                    ASSERT_EQ(pk.data().data()[i], pk2.data().data()[i]);
                }
                ASSERT_TRUE(pk.parms_id() == pk2.parms_id());
            }
        };

        save_load_public_key(scheme_type::bfv);
        save_load_public_key(scheme_type::bgv);
    }
} // namespace sigmatest
