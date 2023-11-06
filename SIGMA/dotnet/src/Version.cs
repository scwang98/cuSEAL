// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System;
using System.IO;
using System.Text;

namespace Microsoft.Research.SIGMA
{
    /// <summary>
    /// This class contains static methods for retrieving Microsoft SIGMA's version numbers.
    /// </summary>
    /// <remark>
    /// Use the name SIGMAVersion to distinguish it from System.Version.
    /// </remark>
    public static class SIGMAVersion
    {
        /// <summary>
        /// Returns Microsoft SIGMA's version number string.
        /// </summary>
        static public string Version => $"{SIGMAVersion.Major}.{SIGMAVersion.Minor}.{SIGMAVersion.Patch}";

        ///
        /// <summary>
        /// Returns Microsoft SIGMA's major version number.
        /// </summary>
        static public byte Major
        {
            get
            {
                NativeMethods.Version_Major(out byte result);
                return result;
            }
        }

        /// <summary>
        /// Returns Microsoft SIGMA's minor version number.
        /// </summary>
        static public byte Minor
        {
            get
            {
                NativeMethods.Version_Minor(out byte result);
                return result;
            }
        }

        /// <summary>
        /// Returns Microsoft SIGMA's patch version number.
        /// </summary>
        static public byte Patch
        {
            get
            {
                NativeMethods.Version_Patch(out byte result);
                return result;
            }
        }
    }
}
