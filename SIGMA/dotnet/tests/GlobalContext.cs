// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using Microsoft.Research.SIGMA;

namespace SIGMANetTest
{
    /// <summary>
    /// Provides a global SIGMAContext that can be used by Tests.
    /// Necessary to run tests fast, as creating a SIGMAContext can take around
    /// 2 seconds.
    /// </summary>
    static class GlobalContext
    {
        static GlobalContext()
        {
            EncryptionParameters encParams = new EncryptionParameters(SchemeType.BFV)
            {
                PolyModulusDegree = 8192,
                CoeffModulus = CoeffModulus.BFVDefault(polyModulusDegree: 8192)
            };
            encParams.SetPlainModulus(65537ul);
            BFVContext = new SIGMAContext(encParams);

            encParams = new EncryptionParameters(SchemeType.CKKS)
            {
                PolyModulusDegree = 8192,
                CoeffModulus = CoeffModulus.BFVDefault(polyModulusDegree: 8192)
            };
            CKKSContext = new SIGMAContext(encParams);

            encParams = new EncryptionParameters(SchemeType.BGV)
            {
                PolyModulusDegree = 8192,
                CoeffModulus = CoeffModulus.BFVDefault(polyModulusDegree: 8192)
            };
            encParams.SetPlainModulus(65537ul);
            BGVContext = new SIGMAContext(encParams);
        }

        public static SIGMAContext BFVContext { get; private set; } = null;
        public static SIGMAContext CKKSContext { get; private set; } = null;
        public static SIGMAContext BGVContext { get; private set; } = null;
    }
}
