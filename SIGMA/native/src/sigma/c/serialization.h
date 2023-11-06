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

SIGMA_C_FUNC Serialization_SIGMAMagic(uint16_t *result);

SIGMA_C_FUNC Serialization_SIGMAHeaderSize(uint8_t *result);

SIGMA_C_FUNC Serialization_IsSupportedComprMode(uint8_t compr_mode, bool *result);

SIGMA_C_FUNC Serialization_ComprModeDefault(uint8_t *result);

SIGMA_C_FUNC Serialization_IsCompatibleVersion(uint8_t *headerptr, uint64_t size, bool *result);

SIGMA_C_FUNC Serialization_IsValidHeader(uint8_t *headerptr, uint64_t size, bool *result);
