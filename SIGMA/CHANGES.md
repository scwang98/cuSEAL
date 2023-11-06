# List of Changes

## Version 4.1.1

### Bug Fixes

- Fixed detection of intrinsics on M1 Macs [(PR #597)](https://github.com/microsoft/SIGMA/pull/597).
- Fixed old dependency version numbers in [README.md](README.md) [(PR #596)](https://github.com/microsoft/SIGMA/pull/596).
- Fixed release date typo in [README.md](README.md).

## Version 4.1.0

### Features

- The BGV scheme now keeps ciphertexts in NTT form. BGV ciphertext multiplication is much faster than version 4.0.0.
- When a BGV ciphertext saved by previous versions is loaded in the current version, it is automatically converted to NTT form.
- Increased `SIGMA_COEFF_MOD_COUNT_MAX`, the maximum number of primes that define the coefficient modulus, from 64 to 256.

### Other Fixes

- Fixed typos [(PR #590)](https://github.com/microsoft/SIGMA/pull/590).
- Added $schema to cgmanifest.json [(PR #558)](https://github.com/microsoft/SIGMA/pull/558).
- Fixed typos [(PR #512)](https://github.com/microsoft/SIGMA/pull/512).
- Fixed typos [(PR #530)](https://github.com/microsoft/SIGMA/pull/530).
- Fixed typos [(PR #509)](https://github.com/microsoft/SIGMA/pull/509).
- Added missing `const` qualifiers [(PR #556)](https://github.com/microsoft/SIGMA/pull/556).
- Added vcpkg installation instructions [(PR #562)](https://github.com/microsoft/SIGMA/pull/562).
- Fixed an issue in specific environments where allocation fails without throwing `std::bad_alloc`.
- Fixed comments (C++) and C/.NET wrapper implementation of an exception thrown by `invariant_noise_budget`.

### Major API Changes

- Added new public methods `mod_reduce_xxx(...)` (native) and `ModReduceXxx(...)` (dotnet) to the class `Evaluator`.

## Version 4.0.0

### Features

- Added BGV scheme [(PR 283)](https://github.com/microsoft/SIGMA/pull/283). Thanks, [Alibaba Gemini Lab](https://alibaba-gemini-lab.github.io/)!
- Added a new example "BGV basics" to native and dotnet.
- Loading objects serialized by Microsoft SIGMA v3.4+ are supported.
- Updated versions of dependencies: GoogleTest from 1.10.0 to 1.11.0 and GoogleBenchmark from 1.5.2 to 1.6.0.

### Other Fixes

- Fixed an ambiguous comment [(PR 375)](https://github.com/microsoft/SIGMA/pull/375).

### Major API Changes

- Added `sigma::scheme_type::bgv`.
- Added a new public method `parms_id()` (native) to the class `EncryptionParameters`.
- Added a new public method `Create(...)` (native and dotnet) with three inputs in the class `CoeffModulus`.
- Added a new public method `correction_factor()` (native) or `CorrectionFactor()` (dotnet) to the class `Ciphertext`.
- Removed the friendship of the class `EncryptionParameters` to the class `SIGMAContext`.

### File Changes

- `native/bench/bgv.cpp` is added.
- Examples are renamed and extended.

## Version 3.7.3

### Features

- All output files including downloaded thirdparty dependencies and Visual Studio project and solution files will be created in the build directory [(PR 427)](https://github.com/microsoft/SIGMA/pull/427).
- Reduced `util::try_minimal_primitive_root` search iterations by half [(PR 430)](https://github.com/microsoft/SIGMA/pull/430). Thanks, [zirconium-n](https://github.com/zirconium-n)!
- Updated .Net SDK version to 6.0.x and supported Visual Studio version to 17 2022.
- Added `SIGMA_AVOID_BRANCHING` option to eleminate branching in critical functions when Microsoft SIGMA is built with maliciously inserted compiler flags.

### Bug Fixes

- Removed exceptions in `KeyGenerator::CreateGaloisKeys` when inputs do not include steps so that even when `EncryptionParameterQualifiers::using_batching` is false Galois automorphisms are still available.

### File Changes

- `dotnet/SIGMANet.sln` is removed.
- `dotnet/SIGMANet.sln.in` is added.

## Version 3.7.2

### Bug Fixes

- Fixed a bug when Intel HEXL is used [(Issue 411)](https://github.com/microsoft/SIGMA/issues/411) [(PR 414)](https://github.com/microsoft/SIGMA/pull/414).
- Fixed an abnormal benchmark case due to AVX512 transitions when Intel HEXL is used [(PR 416)](https://github.com/microsoft/SIGMA/pull/416).

## Version 3.7.1

### Bug Fixes

- Fixed compiler and linker errors in downstream projects when Microsoft SIGMA is built with `SIGMA_BUILD_DEPS=ON` and `SIGMA_USE_INTEL_HEXL=ON`.
- Updated CMake minimum requirement to 3.13.

### File Changes

- `native/src/sigma/util/intel_sigma_ext.h` is removed.
- `native/src/sigma/util/intel_sigma_ext.cpp` is removed.

## Version 3.7.0

### Features

- Improved the performance of `Evaluator::multiply`, `Evaluator::multiply_inplace`, and `Evaluator::square` in the BFV scheme for default parameters with degree `4096` or higher.
- Improved the performance of decryption [(PR 363)](https://github.com/microsoft/SIGMA/pull/363).
- Updated to HEXL version 1.2.1 [(PR 375)](https://github.com/microsoft/SIGMA/pull/375).
- Added more benchmark cases [(PR 379)](https://github.com/microsoft/SIGMA/pull/379).

### Minor API Changes

- `const` methods in `SIGMAContext` and `SIGMAContext::ContextData` classes that used to return a pointer or reference now have a preceeding `const` qualifier.

### Bug Fixes

- Fixed failed tests on PowerPC architecture [(Issue 360)](https://github.com/microsoft/SIGMA/issues/360).

## Version 3.6.6

### Bug Fixes

- Fixed an error when loading seeded ciphertexts serialized by v3.4.x from v3.5.0+.
- Fixed failed tests on ARM64 architecture [(Issue 347)](https://github.com/microsoft/SIGMA/issues/347).

### Other

- Improved HEXL NTT integration [(PR 349)](https://github.com/microsoft/SIGMA/pull/349).
- Improved CKKS ciphertext multiplication [(PR 346)](https://github.com/microsoft/SIGMA/pull/346).
- Improved CKKS ciphertext square [(PR 353)](https://github.com/microsoft/SIGMA/pull/353), except that with GNU G++ compiler and `1024` degree there is a huge penalty in execution time.
Users should switch from GNU G++ in this specific parameter setting if CKKS square is used.

## Version 3.6.5

### New Features

- Updated the dependency Intel HEXL to v1.1.0 [(PR 332)](https://github.com/microsoft/SIGMA/pull/332).
- Integrated more optimizations from Intel HEXL to Microsoft SIGMA.
- Intel HEXL now uses Microsoft SIGMA's memory pool, so that memory allocation reported by Microsoft SIGMA is more accurate.

### Bug Fixes

- Fixed typos in comments [(PR 328)](https://github.com/microsoft/SIGMA/pull/328).
- Fixed a bug in `DWTHandler` [(Issue 330)](https://github.com/microsoft/SIGMA/issues/330).
- Fixed failing tests when `SIGMA_USE_ZLIB=OFF` and `SIGMA_USE_ZTD=OFF` [(PR 332)](https://github.com/microsoft/SIGMA/pull/332).
- Fixed shared library build when `SIGMA_USE_HEXL=ON` [(PR 332)](https://github.com/microsoft/SIGMA/pull/332).
- Added missing `const` qualifiers to several members of `BatchEncoder` and `Evaluator` [(PR 334)](https://github.com/microsoft/SIGMA/pull/334).

## Version 3.6.4

### New Features

- Enabled [AddressSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer) and [LeakSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizerLeakSanitizer) when building Microsoft SIGMA [tests](native/tests/CMakeLists.txt) in Debug mode on Unix-like systems, based on [(PR 318)](https://github.com/microsoft/SIGMA/pull/318).

### Bug Fixes

- Fixed `alloc-dealloc-mismatch` issues resolved by [(PR 318)](https://github.com/microsoft/SIGMA/pull/318).
- Fixed wrong descriptions in [serializable.h](native/src/sigma/serializable.h) and [Serializable.cs](dotnet/src/Serializable.cs) reported in [(Issue 316)](https://github.com/microsoft/SIGMA/issues/316).

## Version 3.6.3

### New Features

- Added support to build Microsoft SIGMA out of the box with [emscripten](https://emscripten.org/) [(PR 306)](https://github.com/microsoft/SIGMA/pull/306).
- Added support to build Microsoft SIGMA with [Intel HEXL](https://github.com/intel/hexl) as an optional dependency [(PR 312)](https://github.com/microsoft/SIGMA/pull/312).

### Other

- Improved the error message when attempting to configure with `BUILD_SHARED_LIBS=ON` and `SIGMA_BUILD_SIGMA_C=ON` [(Issue 284)](https://github.com/microsoft/SIGMA/issues/284).
- Added `sigma::random_bytes` function in [randomgen.h](native/src/sigma/randomgen.h).
- Removed redundant `is_metadata_valid_for` invocations reported in [(Issue 313)](https://github.com/microsoft/SIGMA/issues/313).
- Minor bug fixes

### File Changes

- [cmake/ExternalIntelHEXL.cmake](cmake/ExternalIntelHEXL.cmake)
- [native/src/sigma/util/intel_sigma_ext.h](native/src/sigma/util/intel_sigma_ext.h)

#### New files

## Version 3.6.2

### Hotfix - 2/18/2021

- Merged pull request [(PR 282)](https://github.com/microsoft/SIGMA/pull/282) with typo and minor bug fixes.

### Bug Fixes

- Fixed an issue [(Issue 278)](https://github.com/microsoft/SIGMA/issues/278) in finding ZLIB header files when building SIGMA with `BUILD_SHARED_LIBS=ON`.
- Fixed a member variable initialization order bug in [SafeByteBuffer](native/src/sigma/util/streambuf.h).

### New Features

- Added benchmarks that depend on Google Benchmark in [native/bench](native/bench).

### Other

- Changed low-level code that reduces the runtime difference among code generated by msvc, gcc, and clang.
- Using ARM64 intrinsics for better performance [(PR 269)](https://github.com/microsoft/SIGMA/pull/269).

## Version 3.6.1

- Fixed a bug reported in [(Issue 248)](https://github.com/microsoft/SIGMA/issues/248) and [(Issue 249)](https://github.com/microsoft/SIGMA/issues/249): in in-place Zstandard compression the input buffer head location was not correctly updated, resulting in huge memory use.

## Version 3.6.0

### Hotfix - 12/2/2020

- Fixed an issue with CMake system where a shared Zstandard was not correctly handled (it is not supported).

### Hotfix - 11/17/2020

- Fixed an issue with CMake system where `BUILD_SHARED_LIBS=ON` and `SIGMA_BUILD_DEPS=ON` resulted in Zstandard header files not being visible to the build [(Issue 242)](https://github.com/microsoft/SIGMA/issues/242).

### Hotfix - 11/16/2020

- Fixed issues with CMake system overwriting existing `FETCHCONTENT_BASE_DIR` [(Issue 242)](https://github.com/microsoft/SIGMA/issues/242).
- Corrected mistakes and typos in [README.md](README.md).

### New Features

- Added support for [Zstandard](https://github.com/facebook/zstd) compression as a much more efficient alternative to ZLIB.
The performance improvement is around 20&ndash;30x.
- Added support for iOS in the [NuGet package of Microsoft SIGMA](https://www.nuget.org/packages/Microsoft.Research.SIGMANet).
- The build system is unified for all platforms.
There is no longer a Visual Studio solution file (`sigma.sln`) for Windows.
There is a separate solution file for the dotnet library ([dotnet/SIGMANet.sln](dotnet/SIGMANet.sln)).
- Added support for Shake256 (FIPS-202) XOF for pseudo-random number generation in addition to the default Blake2xb (faster).
- Microsoft SIGMA 3.6 is backwards compatible with 3.4 and 3.5 when deserializing, but it does not support serializing in the old formats.

### Major API Changes

- All C++ `enum` labels are consistently in lowercase. Most importantly, `scheme_type::BFV` and `scheme_type::CKKS` are changed to `scheme_type::bfv` and `scheme_type::ckks`.
- Changed `sigma::SIGMA_BYTE` to `sigma::sigma_byte`; all uppercase names are used only for preprocessor macros.
- Removed `BatchEncoder` API for encoding and decoding `Plaintext` objects inplace.
This is because a `Plaintext` object with slot-data written into the coefficients is (confusingly) not valid to be used for encryption.
- Removed `IntegerEncoder` and `BigUInt` classes.
`IntegerEncoder` results in inefficient homomorphic evaluation and lacks sane correctness properties, so it was basically impossible to use in real applications.
The `BigUInt` class was only used by the `IntegerEncoder`.
- All `Encryptor::encrypt` variants have now two overloads: one that takes a `Ciphertext` out-parameter, and one that returns a `Serializable<Ciphertext>`.
- Changed the names of the public key generation functions to clearly express that a new key is created each time, e.g., `KeyGenerator::create_public_key`.
- Removed the `KeyGenerator::relin_keys_local` and `KeyGenerator::galois_keys_local` functions.
These were poorly named and have been replaced with overloads of `KeyGenerator::create_relin_keys` and `KeyGenerator::create_galois_keys` that take an out-parameter of type `RelinKeys` or `GaloisKeys`.
- Renamed `IntArray` to `DynArray` (dynamic array) and removed unnecessary limitations on the object type template parameter.
- Added public API for modular reduction to the `Modulus` class.
- Added API for creating `DynArray` and `Plaintext` objects from a `gsl::span<std::uint64_t>` (C++) or `IEnumerable<ulong>` (C#).

### Minor API Changes

- Added `std::hash` implementation for `EncryptionParameters` (in addition to `parms_id_type`) so it is possible to create e.g. `std::unordered_map` of `EncryptionParameters`.
- Added API to `UniformRandomGeneratorFactory` to find whether the factory uses a default seed and to retrieve that seed.
- Added const overloads for `DynArray::begin` and `DynArray::end`.
- Added a `Shake256PRNG` and `Shake256PRNGFactory` classes.
Renamed `BlakePRNG` class to `Blake2xbPRNG`, and `BlakePRNGFactory` class to `Blake2xbPRNGFactory`.
- Added a serializable `UniformRandomGeneratorInfo` class that represents the type of an extendable output function and a seed value.
- Added [native/src/sigma/version.h](native/src/sigma/version.h) defining a struct `SIGMAVersion`.
This is used internally to route deserialization logic to correct functions depending on loaded `SIGMAHeader` version.

### New Build Options

- `SIGMA_BUILD_DEPS` controls whether dependencies are downloaded and built into Microsoft SIGMA or searched from the system.
- Only a shared library will be built when `BUILD_SHARED_LIBS` is set to `ON`. Previously a static library was always built.
- Encryption error is sampled from a Centered Binomial Distribution (CBD) by default unless `SIGMA_USE_GAUSSIAN_NOISE` is set to `ON`.
Sampling from a CBD is constant-time and faster than sampling from a Gaussian distribution, which is why it is used by many of the NIST PQC finalists.
- `SIGMA_DEFAULT_PRNG` controls which XOF is used for pseudo-random number generation.
The available values are `Blake2xb` (default) and `Shake256`.

### Other

- The pkg-config system has been improved.
All files related to pkg-config have been moved to [pkgconfig/](pkgconfig/).
CMake creates now also a pkg-config file `sigma_shared.pc` for compiling against a shared Microsoft SIGMA if `BUILD_SHARED_LIBS` is set to `ON`.
- Added `.pre-commit-config.yaml` (check out [pre-commit](https://pre-commit.com) if you are not familiar with this tool).
- Added `sigma::util::DWTHandler` and `sigma::util::Arithmetic` class templates that unify the implementation of FFT (used by `CKKSEncoder`) and NTT (used by polynomial arithmetic).
- The performance of encoding and decoding in CKKS are improved.
- The performance of randomness generation for ciphertexts and keys (RLWE samples) is improved.

### File Changes

#### Renamed files and directories

- `native/src/sigma/intarray.h` to [native/src/sigma/dynarray.h](native/src/sigma/dynarray.h)
- `dotnet/src/SIGMANet.csproj` to [dotnet/src/SIGMANet.csproj.in](dotnet/src/SIGMANet.csproj.in)
- `dotnet/tests/SIGMANetTest.csproj` to [dotnet/tests/SIGMANetTest.csproj.in](dotnet/tests/SIGMANetTest.csproj.in)
- `dotnet/examples/SIGMANetExamples.csproj` to [dotnet/examples/SIGMANetExamples.csproj.in](dotnet/examples/SIGMANetExamples.csproj.in)

#### New files

- [native/src/sigma/util/dwthandler.h](native/src/sigma/util/dwthandler.h)
- [native/src/sigma/util/fips202.h](native/src/sigma/util/fips202.h)
- [native/src/sigma/util/fips202.c](native/src/sigma/util/fips202.c)
- [native/src/sigma/version.h](native/src/sigma/version.h)
- [dotnet/SIGMANet.sln](dotnet/SIGMANet.sln)
- [.pre-commit-config.yaml](.pre-commit-config.yaml)

#### Removed files

- `dotnet/src/BigUInt.cs`
- `dotnet/src/IntegerEncoder.cs`
- `dotnet/tests/BigUIntTests.cs`
- `dotnet/tests/IntegerEncoderTests.cs`
- `native/examples/SIGMAExamples.vcxproj`
- `native/examples/SIGMAExamples.vcxproj.filters`
- `native/src/CMakeConfig.cmd`
- `native/src/SIGMA_C.vcxproj`
- `native/src/SIGMA_C.vcxproj.filters`
- `native/src/SIGMA.vcxproj`
- `native/src/SIGMA.vcxproj.filters`
- `native/src/sigma/biguint.h`
- `native/src/sigma/biguint.cpp`
- `native/src/sigma/intencoder.h`
- `native/src/sigma/intencoder.cpp`
- `native/tests/packages.config`
- `native/tests/SIGMATest.vcxproj`
- `native/tests/SIGMATest.vcxproj.filters`
- `native/tests/sigma/biguint.cpp`
- `native/tests/sigma/intencoder.cpp`
- `thirdparty/`
- `SIGMA.sln`

## Version 3.5.9

### Bug fixes

- Fixed [(Issue 216)](https://github.com/microsoft/SIGMA/issues/216).
- Fixed [(Issue 210)](https://github.com/microsoft/SIGMA/issues/210).

## Version 3.5.8

### Other

- The bug fixed in [(PR 209)](https://github.com/microsoft/SIGMA/pull/209) also affects Android. Changed version to 3.5.8 where this is fixed.

## Version 3.5.7

### Hotfix - 8/28/2020

- Merged [(PR 209)](https://github.com/microsoft/SIGMA/pull/209). Thanks [s0l0ist](https://github.com/s0l0ist)!

### Bug fixes

- Fixed an omission in input validation in decryption: the size of the ciphertext was not checked to be non-zero.

### Other

- In Windows switch to using `RtlGenRandom` if the BCrypt API fails.
- Improved performance in serialization: data clearing memory pools were always used before, but now are only used for the secret key.
- Use native APIs for memory clearing, when available, instead of for-loop.

## Version 3.5.6

### Bug fixes

- Fixed a bug where setting a PRNG factory to use a constant seed did not result in deterministic ciphertexts or public keys.
The problem was that the specified PRNG factory was not used to sample the uniform part of the RLWE sample(s), but instead a fresh (secure) PRNG was always created and used.
- Fixed a bug where the `parms_id` of a `Plaintext` was not cleared correctly before resizing in `Decryptor::bfv_decrypt`.
As a result, a plaintext in NTT form could not be used as the destination for decrypting a BFV ciphertext.

### Other

- Merged pull request [(Issue 190)](https://github.com/microsoft/SIGMA/pull/190) to replace global statics with function-local statics to avoid creating these objects unless they are actually used.

## Version 3.5.5

### Hotfix - 7/6/2020

- Fixed [(Issue 188)](https://github.com/microsoft/SIGMA/issues/188).

### New features

- Added a struct `sigma::util::MultiplyUIntModOperand` in [native/src/sigma/util/uintarithsmallmod.h](native/src/sigma/util/uintarithsmallmod.h).
This struct handles precomputation data for Barrett style modular multiplication.
- Added new overloads for modular arithmetic in [native/src/sigma/util/uintarithsmallmod.h](native/src/sigma/util/uintarithsmallmod.h) where one operand is replaced by a `MultiplyUIntModOperand` instance for improved performance when the same operand is used repeatedly.
- Changed the name of `sigma::util::barrett_reduce_63` to `sigma::util::barrett_reduce_64`; the name was misleading and only referred to the size of the modulus.
- Added `sigma::util::StrideIter` in [native/src/sigma/util/iterator.h](native/src/sigma/util/iterator.h).
- Added macros `SIGMA_ALLOCATE_GET_PTR_ITER` and `SIGMA_ALLOCATE_GET_STRIDE_ITER` in [native/src/sigma/util/defines.h](native/src/sigma/util/defines.h).

### Other

- Significant performance improvements from merging pull request [(PR 185)](https://github.com/microsoft/SIGMA/pull/185) and implementing other improvements of the same style (see above).
- Removed a lot of old and unused code.

## Version 3.5.4

### Bug Fixes

- `std::void_t` was introduced only in C++17; switched to using a custom implementation [(Issue 180)](https://github.com/microsoft/SIGMA/issues/180).
- Fixed two independent bugs in `native/src/CMakeConfig.cmd`: The first prevented SIGMA to be built in a directory with spaces in the path due to missing quotation marks. Another issue caused MSVC to fail when building SIGMA for multiple architectures.
- `RNSBase::decompose_array` had incorrect semantics that caused `Evaluator::multiply_plain_normal` and `Evaluator::transform_to_ntt_inplace` (for `Plaintext`) to behave incorrectly for some plaintexts.

### Other

- Added pkg-config support [(PR 181)](https://github.com/microsoft/SIGMA/pull/181).
- `sigma::util::PtrIter<T *>` now dereferences correctly to `T &` instead of `T *`.
This results in simpler code, where inside `SIGMA_ITERATE` lambda functions dereferences of `sigma::util::PtrIter<T *>` do not need to be dereferenced a second time, as was particularly common when iterating over `ModulusIter` and `NTTTablesIter` types.
- `sigma::util::IterTuple` now dereferences to an `std::tuple` of dereferences of its component iterators, so it is no longer possible to directly pass a dereferenced `sigma::util::IterTuple` to an inner lambda function in nested `SIGMA_ITERATE` calls.
Instead, the outer lambda function parameter should be wrapped inside another call to `sigma::util::iter` before passed on to the inner `SIGMA_ITERATE` to produce an appropriate `sigma::util::IterTuple`.

## Version 3.5.3

### Bug Fixes

- Fixed a bug in `sigma::util::IterTuple<...>` where a part of the `value_type` was constructed incorrectly.
- Fixed a bug in `Evaluator::mod_switch_drop_to_next` that caused non-inplace modulus switching to fail [(Issue 179)](https://github.com/microsoft/SIGMA/issues/179). Thanks s0l0ist!

## Version 3.5.2

### Bug Fixes

- Merged pull request [PR 178](https://github.com/microsoft/SIGMA/pull/178) to fix a lambda capture issue when building on GCC 7.5.
- Fixed issue where SIGMA.vcxproj could not be compiled with MSBuild outside the solution [(Issue 171)](https://github.com/microsoft/SIGMA/issues/171).
- SIGMA 3.5.1 required CMake 3.13 instead of 3.12; this has now been fixed and 3.12 works again [(Issue 167)](https://github.com/microsoft/SIGMA/issues/167).
- Fixed issue in NuSpec file that made local NuGet package generation fail.
- Fixed issue in NuSpec where XML documentation was not included into the package.

### New Features

- Huge improvements to SIGMA iterators, including `sigma::util::iter` and `sigma::util::reverse_iter` functions that can create any type of iterator from appropriate parameters.
- Added `sigma::util::SeqIter<T>` iterator for iterating a sequence of numbers for convenient iteration indexing.
- Switched functions in `sigma/util/polyarithsmallmod.*` to use iterators; this is to reduce the layers of iteration in higher level code.
- Added macro `SIGMA_ITERATE` that should be used instead of `for_each_n`.

### Other

- Added note in [README.md](README.md) about known performance issues when compiling with GNU G++ compared to Clang++ [(Issue 173)](https://github.com/microsoft/SIGMA/issues/173).
- Merged pull requests that improve the performance of keyswitching [(PR 177)](https://github.com/microsoft/SIGMA/pull/177) and rescale [(PR 176)](https://github.com/microsoft/SIGMA/pull/176) in CKKS.

## Version 3.5.1

Changed version to 3.5.1. The two hotfixes below are included.

## Version 3.5.0

### Hotfix - 4/30/2020

- Fixed a critical bug [(Issue 166)](https://github.com/microsoft/SIGMA/issues/166) in `Evaluator::multiply_plain_inplace`. Thanks s0l0ist!

### Hotfix - 4/29/2020

- Switched to using Microsoft GSL v3.0.1 and fixed minor GSL related issues in `CMakeLists.txt`.
- Fixed some typos in [README.md](README.md).
- Fixes bugs in ADO pipelines files.

### New Features

- Microsoft SIGMA officially supports Android (Xamarin.Android) on ARM64.
- Microsoft SIGMA is a CMake project (UNIX-like systems only):
  - There is now a top-level `CMakeLists.txt` that builds all native components.
  - The following CMake targets are created: `SIGMA::sigma` (static library), `SIGMA::sigma_shared` (shared library; optional), `SIGMA::sigmac` (C export library; optional).
  - Examples and unit tests are built if enabled through CMake (see [README.md](README.md)).
  - ZLIB is downloaded and compiled by CMake and automatically included in the library.
  - Microsoft GSL is downloaded by CMake. Its header files are copied to `native/src/gsl` and installed with Microsoft SIGMA.
  - Google Test is downloaded and compiled by CMake.
- Improved serialization:
  - `Serialization::SIGMAHeader` layout has been changed. SIGMA 3.4 objects can still be loaded by SIGMA 3.5, and the headers are automatically converted to SIGMA 3.5 format.
  - `Serialization::SIGMAHeader` captures version number information.
  - Added examples for serialization.
  - The seeded versions of `Encryptor`'s symmetric-key encryption and `KeyGenerator`'s `RelinKeys` and `GaloisKeys` generation now output `Serializable` objects. See more details in *API Changes* below.

#### For Library Developers and Contributors

We have created a set of C++ iterators that easily allows looping over polynomials in a ciphertext, over RNS components in a polynomial, and over coefficients in an RNS component.
There are also a few other iterators that can come in handy.
Currently `Evaluator` fully utilizes these, and in the future the rest of the library will as well.
The iterators are primarily intended to be used with `std::for_each_n` to simplify existing code and help with code correctness.
Please see [native/src/sigma/util/iterator.h](native/src/sigma/util/iterator.h) for guidance on how to use these.

We have also completely rewritten the RNS tools that were previously in the `util::BaseConverter` class.
This functionality is now split between two classes: `util::BaseConverter` whose sole purpose is to perform the `FastBConv` computation of [[BEHZ16]](https://eprint.iacr.org/2016/510) and `util::RNSTool` that handles almost everything else.
RNS bases are now represented by the new `util::RNSBase` class.

### API Changes

The following changes are explained in C++ syntax and are introduced to .NET wrappers similarly:

- New generic class `Serializable` wraps `Ciphertext`, `RelinKeys`, and `GaloisKeys` objects to provide a more flexible approach to the functionality provided in release 3.4 by `KeyGenerator::[relin|galois]_keys_save` and `Encryptor::encrypt_[zero_]symmetric_save` functions.
Specifically, these functions have been removed and replaced with overloads of `KeyGenerator::[relin|galois]_keys` and `Encryptor::encrypt_[zero_]symmetric` that return `Serializable` objects.
The `KeyGenerator::[relin|galois]_keys` methods in release 3.4 are renamed to `KeyGenerator::[relin|galois]_keys_local`.
The `Serializable` objects cannot be used directly by the API, and are only intended to be serialized, which activates the compression functionalities introduced earlier in release 3.4.
- `SmallModulus` class is renamed to `Modulus`, and is relocated to [native/src/sigma/modulus.h](native/src/sigma/modulus.h).
- `*coeff_mod_count*` methods are renamed to `*coeff_modulus_size*`, which applies to many classes.
- `parameter_error_name` and `parameter_error_message` methods are added to `EncryptionParameterQualifiers` and `SIGMAContext` classes to explain why an `EncryptionParameters` object is invalid.
- The data members and layout of `Serialization::SIGMAHeader` have changed.

The following changes are specific to C++:

- New bounds in [native/src/sigma/util/defines.h](native/src/sigma/util/defines.h):
  - `SIGMA_POLY_MOD_DEGREE_MAX` is increased to 131072; values bigger than 32768 require the security check to be disabled by passing `sec_level_type::none` to `SIGMAContext::Create`.
  - `SIGMA_COEFF_MOD_COUNT_MAX` is increased to 64.
  - `SIGMA_MOD_BIT_COUNT_MAX` and `SIGMA_MOD_BIT_COUNT_MIN` are added and set to 61 and 2, respectively.
  - `SIGMA_INTERNAL_MOD_BIT_COUNT` is added and set to 61.
- `EncryptionParameterQualifiers` now has an error code `parameter_error` that interprets the reason why an `EncryptionParameter` object is invalid.
- `bool parameters_set()` is added to replace the previous `bool parameters_set` member.

The following changes are specific to .NET:

- Version numbers are retrievable in .NET through `SIGMAVersion` class.

### Other Changes

- Releases are now listed on [releases page](https://github.com/microsoft/SIGMA/releases).
- The native library can serialize (save and load) objects larger than 4 GB.
Please be aware that compressed serialization requires an additional temporary buffer roughly the size of the object to be allocated, and the streambuffer for the output stream may consume some non-trivial amount of memory as well.
In the .NET library, objects are limited to 2 GB, and loading an object larger than 2 GB will throw an exception.
[(Issue 142)](https://github.com/microsoft/SIGMA/issues/142)
- Larger-than-suggested parameters are supported for expert users.
To enable that, please adjust `SIGMA_POLY_MOD_DEGREE_MAX` and `SIGMA_COEFF_MOD_COUNT_MAX` in [native/src/sigma/util/defines.h](native/src/sigma/util/defines.h).
([Issue 150](https://github.com/microsoft/SIGMA/issues/150),
[Issue 84](https://github.com/microsoft/SIGMA/issues/84))
- Serialization now clearly indicates an insufficient buffer size error.
[(Issue 117)](https://github.com/microsoft/SIGMA/issues/117)
- Unsupported compression mode now throws `std::invalid_argument` (native) or `ArgumentException` (.NET).
- There is now a `.clang-format` for automated formatting of C++ (`.cpp` and `.h`) files.
Execute `tools/scripts/clang-format-all.sh` for easy formatting (UNIX-like systems only).
This is compatible with clang-format-9 and above.
Formatting for C# is not yet supported.
[(Issue 93)](https://github.com/microsoft/SIGMA/issues/93)
- The C export library previously in `dotnet/native/` is moved to [native/src/sigma/c/](native/src/sigma/c/) and renamed to SIGMA_C to support building of wrapper libraries in languages like .NET, Java, Python, etc.
- The .NET wrapper library targets .NET Standard 2.0, but the .NET example and test projects use C# 8.0 and require .NET Core 3.x. Therefore, Visual Studio 2017 is no longer supported for building the .NET example and test projects.
- Fixed issue when compiling in FreeBSD.
([PR 113](https://github.com/microsoft/SIGMA/pull/113))
- A [bug](https://eprint.iacr.org/2019/1266) in the [[BEHZ16]](https://eprint.iacr.org/2016/510)-style RNS operations is fixed; proper unit tests are added.
- Performance of methods in `Evaluator` are in general improved.
([PR 148](https://github.com/microsoft/SIGMA/pull/148))
This is compiler-dependent, however, and currently Clang seems to produce the fastest running code for Microsoft SIGMA.

### File Changes

Renamed files and directories:

- [dotnet/examples/7_Performance.cs](dotnet/examples/7_Performance.cs) was previously `dotnet/examples/6_Performance.cs`
- [native/examples/7_performance.cpp](native/examples/7_performance.cpp) was previously `native/examples/6_performance.cpp`
- [native/src/sigma/c/](native/src/sigma/c/) was previously `dotnet/native/sigmanet`.
- [native/src/sigma/util/ntt.h](native/src/sigma/util/ntt.h) was previously `native/src/sigma/util/smallntt.h`.
- [native/src/sigma/util/ntt.cpp](native/src/sigma/util/ntt.cpp) was previously `native/src/sigma/util/smallntt.cpp`.
- [native/tests/sigma/util/ntt.cpp](native/tests/sigma/util/ntt.cpp) was previously `native/tests/sigma/util/smallntt.cpp`.

New files:

- [android/](android/)
- [dotnet/examples/6_Serialization.cs](dotnet/examples/6_Serialization.cs)
- [dotnet/src/Serializable.cs](dotnet/src/Serializable.cs)
- [dotnet/src/Version.cs](dotnet/src/Version.cs)
- [dotnet/tests/SerializationTests.cs](dotnet/tests/SerializationTests.cs)
- [native/examples/6_serialization.cpp](native/examples/6_serialization.cpp)
- [native/src/sigma/c/version.h](native/src/sigma/c/version.h)
- [native/src/sigma/c/version.cpp](native/src/sigma/c/version.cpp)
- [native/src/sigma/util/galois.h](native/src/sigma/util/galois.h)
- [native/src/sigma/util/galois.cpp](native/src/sigma/util/galois.cpp)
- [native/src/sigma/util/hash.cpp](native/src/sigma/util/hash.cpp)
- [native/src/sigma/util/iterator.h](native/src/sigma/util/iterator.h)
- [native/src/sigma/util/rns.h](native/src/sigma/util/rns.h)
- [native/src/sigma/util/rns.cpp](native/src/sigma/util/rns.cpp)
- [native/src/sigma/util/streambuf.h](native/src/sigma/util/streambuf.h)
- [native/src/sigma/util/streambuf.cpp](native/src/sigma/util/streambuf.cpp)
- [native/src/sigma/serializable.h](native/src/sigma/serializable.h)
- [native/tests/sigma/util/iterator.cpp](native/tests/sigma/util/iterator.cpp)
- [native/tests/sigma/util/galois.cpp](native/tests/sigma/util/galois.cpp)
- [native/tests/sigma/util/rns.cpp](native/tests/sigma/util/rns.cpp)

Removed files:

- `dotnet/src/SmallModulus.cs` is merged to [dotnet/src/ModulusTests.cs](dotnet/src/Modulus.cs).
- `dotnet/tests/SmallModulusTests.cs` is merged to [dotnet/tests/ModulusTests.cs](dotnet/tests/ModulusTests.cs).
- `native/src/sigma/util/baseconverter.h`
- `native/src/sigma/util/baseconverter.cpp`
- `native/src/sigma/smallmodulus.h` is merged to [native/src/sigma/modulus.h](native/src/sigma/modulus.h).
- `native/src/sigma/smallmodulus.cpp` is merged to [native/src/sigma/modulus.cpp](native/src/sigma/modulus.cpp).
- `native/src/sigma/c/smallmodulus.h` is merged to [native/src/sigma/c/modulus.h](native/src/sigma/c/modulus.h).
- `native/src/sigma/c/smallmodulus.cpp` is merged to [native/src/sigma/c/modulus.cpp](native/src/sigma/c/modulus.cpp).
- `native/tests/sigma/smallmodulus.cpp` is merged to [native/tests/sigma/modulus.cpp](native/tests/sigma/modulus.cpp).
- `native/tests/sigma/util/baseconverter.cpp`

## Version 3.4.5

- Fixed a concurrency issue in SIGMANet: the `unordered_map` storing `SIGMAContext` pointers was not locked appropriately on construction and destruction of new `SIGMAContext` objects.
- Fixed a few typos in examples ([PR 71](https://github.com/microsoft/SIGMA/pull/71)).
- Added include guard to config.h.in.

## Version 3.4.4

- Fixed issues with `SIGMANet.targets` file and `SIGMANet.nuspec.in`.
- Updated `README.md` with information about existing multi-platform [NuGet package](https://www.nuget.org/packages/Microsoft.Research.SIGMANet).

## Version 3.4.3

- Fixed bug in .NET serialization code where an incorrect number of bytes was written when using ZLIB compression.
- Fixed an issue with .NET functions `Encryptor.EncryptSymmetric...`, where asymmetric encryption was done instead of symmetric encryption.
- Prevented `KeyGenerator::galois_keys` and `KeyGenerator::relin_keys` from being called when the encryption parameters do not support keyswitching.
- Fixed a bug in `Decryptor::invariant_noise_budget` where the computed noise budget was `log(plain_modulus)` bits smaller than it was supposed to be.
- Removed support for Microsoft GSL `gsl::multi_span`, as it was recently deprecated in GSL.

## Version 3.4.2

- Fixed bug reported in [Issue 66](https://github.com/microsoft/SIGMA/issues/66) on GitHub.
- CMake does version matching now (correctly) only on major and minor version, not patch version, so writing `find_package(SIGMA 3.4)` works correctly and selects the newest version `3.4.x` it can find.

## Version 3.4.1

This patch fixes a few issues with ZLIB support on Windows.
Specifically,

- Fixed a mistake in `native/src/CMakeConfig.cmd` where the CMake library search path
suffix was incorrect.
- Switched to using a static version of ZLIB on Windows.
- Corrected instructions in [README.md](README.md) for enabling ZLIB support on Windows.

## Version 3.4.0

### New Features

- Microsoft SIGMA can use [ZLIB](https://github.com/madler/zlib), a data compression library, to automatically compress data that is serialized. This applies to every serializable object in Microsoft SIGMA.
This feature must be enabled by the user. See more explanation of the compression mechanism in [README.md](README.md#zlib).
Microsoft SIGMA does not redistribute ZLIB.
- AES-128 is replaced with the BLAKE2 family of hash functions in the pseudorandom number generator, as BLAKE2 provides better cross-platform support.
Microsoft SIGMA redistributes the [reference implementation of BLAKE2](https://github.com/BLAKE2/BLAKE2) with light modifications to silence some misleading warnings in Visual Studio.
The reference implementation of BLAKE2 is licensed under [CC0 1.0 Universal](https://github.com/BLAKE2/BLAKE2/blob/master/COPYING); see license boilerplates in files [native/src/sigma/util/blake*](native/src/sigma/util/).
- The serialization functionality has been completely rewritten to make it more safe and robust.
Every serialized Microsoft SIGMA object starts with a 16-byte `Serialization::SIGMAHeader` struct, and then includes the data for the object member variables.
Every serializable object can now also be directly serialized into a memory buffer instead of a C++ stream.
This improves serialization for .NET and makes it much easier to wrap the serialization functionality in other languages, e.g., Java.
Unfortunately, old serialized Microsoft SIGMA objects are incompatible with the new format.
- A ciphertext encrypted with a secret key, for example, a keyswitching key, has one component generated by the PRNG.
By using a seeded PRNG, this component can be replaced with the random seed used by the PRNG to reduce data size.
After transmitted to another party with Microsoft SIGMA, the component can be restored (regenerated) with the same seed.
The security of using seeded PRNG is enhanced by switching to BLAKE2 hash function with a 512-bit seed.
- `Encryptor` now can be constructed with a secret key.
This enables symmetric key encryption which has methods that serialize ciphertexts (compressed with a seed) to a C++ stream or a memory buffer.
- The CMake system has been improved.
For example, multiple versions of Microsoft SIGMA can now be installed on the same system easily, as the default installation directory and library filename now depend on the version of Microsoft SIGMA.
Examples and unit tests can now be built without installing the library.
[README.md](README.md) has been updated to reflect these changes.
- `Encryptor::encrypt` operations in the BFV scheme are modified.
Each coefficient of a plaintext message is first multiplied with the ciphertext modulus, then divided by the plaintext modulus, and rounded to the nearest integer.
In comparison with the previous method, where each coefficient of a plaintext message is multiplied with the flooring of the coefficient modulus divided by the plaintext modulus, the new method reduces the noise introduced in encryption, increases a noise budget of a fresh encryption, slightly slows down encryption, and has no impact on the security at all.
- Merged [PR 62](https://github.com/microsoft/SIGMA/pull/62) that uses a non-adjacent form (NAF) decomposition of random rotations to perform them in a minimal way from power-of-two rotations in both directions.
- This improves performance of random rotations.

### API Changes

#### C++ Native

In all classes with `save` and `load` methods:

- Replaced the old `save` with two new methods that saves to either a C++ stream or a memory buffer.
Optionally, a compression mode can be chosen when saving an object.
- Replaced the old `load` with two new methods that loads from either a C++ stream or a memory buffer.
- Added a method `save_size` to get an upper bound on the size of the object as if it was written to an output stream.
To save to a buffer, the user must ensure that the buffer has at least size equal to what the `save_size` member function returns.
- New `save` and `load` methods rely on the `Serialization` class declared in `serialization.h`.
This class unifies the serialization functionality for all serializable Microsoft SIGMA classes.

In class `Ciphertext`:

- Added a method `int_array` for read-only access to the underlying `IntArray` object.
- Removed methods `uint64_count_capacity` and `uint64_count` that can now be accessed in a more descriptive manner through the `int_array` return value.

In class `CKKSEncoder`: added support for `gsl::span` type of input.

In class `SIGMAContext::ContextData`: added method `coeff_mod_plain_modulus` for read-only access to the non-RNS version of `upper_half_increment`.

In class `EncryptionParameters`: an `EncryptionParameters` object can be constructed without `scheme_type` which by default is set to `scheme_type::none`.

In class `Encryptor`:

- An `Encryptor` object can now be constructed with a secret key to enable symmetric key encryption.
- Added methods `encrypt_symmetric` and `encrypt_zero_symmetric` that generate a `Ciphertext` using the secret key.
- Added methods `encrypt_symmetric_save` and `encrypt_zero_symmetric_save` that directly serialize the resulting `Ciphertext` to a C++ stream or a memory buffer.
The resulting `Ciphertext` no long exists after serilization.
In these methods, the second polynomial of a ciphertext is generated by the PRNG and is replaced with the random seed used.

In class `KeyGenerator`:

- Added methods `relin_keys_save` and `galois_keys_save` that generate and directly serialize keys to a C++ stream or a memory buffer.
The resulting keys no long exist after serilization.
In these methods, half of the polynomials in keys are generated by the PRNG and is replaced with the random seed used.
- Methods `galois_keys` and `galois_keys_save` throw an exception if `EncryptionParameters` do not support batching in the BFV scheme.

In class `Plaintext`: added a method `int_array` for read-only access to the underlying `IntArray` object.

In class `UniformRandomGenerator` and `UniformRandomGeneratorFactory`: redesigned for users to implement their own random number generators more easily.

In file `valcheck.h`: validity checks are partitioned into finer methods; the `is_valid_for(...)` functions will validate all aspects fo the Microsoft SIGMA ojects.

New classes `BlakePRNG` and `BlakePRNGFactory`: uses Blake2 family of hash functions for PRNG.

New class `Serialization`:

- Gives a uniform serialization in Microsoft SIGMA to save objects to a C++ stream or a memory buffer.
- Can be configured to use ZLIB compression.

New files:

- [native/src/sigma/util/rlwe.h](native/src/sigma/util/rlwe.h)
- [native/src/sigma/util/blake2.h](native/src/sigma/util/blake2.h)
- [native/src/sigma/util/blake2-impl.h](native/src/sigma/util/blake2-impl.h)
- [native/src/sigma/util/blake2b.c](native/src/sigma/util/blake2b.c)
- [native/src/sigma/util/blake2xb.c](native/src/sigma/util/blake2xb.c)
- [native/src/sigma/util/croots.cpp](native/src/sigma/util/croots.cpp)
- [native/src/sigma/util/croots.h](native/src/sigma/util/croots.h)
- [native/src/sigma/util/scalingvariant.cpp](native/src/sigma/util/scalingvariant.cpp)
- [native/src/sigma/util/scalingvariant.h](native/src/sigma/util/scalingvariant.h)
- [native/src/sigma/util/ztools.cpp](native/src/sigma/util/ztools.cpp)
- [native/src/sigma/util/ztools.h](native/src/sigma/util/ztools.h)
- [native/src/sigma/serialization.cpp](native/src/sigma/serialization.cpp)
- [native/src/sigma/serialization.h](native/src/sigma/serialization.h)
- [native/tests/sigma/serialization.cpp](native/tests/sigma/serialization.cpp)
- [dotnet/native/sigmanet/serialization_wrapper.cpp](dotnet/native/sigmanet/serialization_wrapper.cpp)
- [dotnet/native/sigmanet/serialization_wrapper.h](dotnet/native/sigmanet/serialization_wrapper.h)

Removed files:

- [native/src/sigma/util/hash.cpp](native/src/sigma/util/hash.cpp)

#### .NET

API changes are mostly identical in terms of functionality to those in C++ native, except only the `IsValidFor` variant of the validity check functions is available in .NET, the more granular checks are not exposed.

New files:

- [dotnet/src/Serialization.cs](dotnet/src/Serialization.cs)

### Minor Bug and Typo Fixes

- Function `encrypt_zero_asymmetric` in [native/src/sigma/util/rlwe.h](native/src/sigma/util/rlwe.h) handles the condition `is_ntt_form == false` correctly.
- Invariant noise calculation in BFV is now correct when the plaintext modulus is large and ciphertexts are fresh (reported in [issue 59](https://github.com/microsoft/SIGMA/issues/59)).
- Fixed comments in [native/src/sigma/util/smallntt.cpp](native/src/sigma/util/smallntt.cpp) as reported in [issue 56](https://github.com/microsoft/SIGMA/issues/56).
- Fixed an error in examples as reported in [issue 61](https://github.com/microsoft/SIGMA/issues/61).
- `GaloisKeys` can no longer be created with encryption parameters that do not support batching.
- Security issues in deserialization were resolved.

## Version 3.3.2 (patch)

### Minor Bug and Typo Fixes

- Switched to using RNS rounding instead of RNS flooring to fix the CKKS
accuracy issue reported in [issue 52](https://github.com/microsoft/SIGMA/issues/52).
- Added support for QUIET option in CMake (`find_package(sigma QUIET)`).
- Using `[[nodiscard]]` attribute when compiling as C++17.
- Fixed a bug in `Evaluator::multiply_many` where the input vector was changed.

## Version 3.3.1 (patch)

### Minor Bug and Typo Fixes

- A bug was fixed that introduced significant extra inaccuracy in CKKS when compiled on Linux, at least with some versions of glibc; Windows and macOS were not affected.
- A bug was fixed where, on 32-bit platforms, some versions of GCC resolved the util::reverse_bits function to the incorrect overload.

## Version 3.3.0

### New Features

In this version, we have significantly improved the usability of the CKKS scheme in Microsoft SIGMA and many of these improvements apply to the BFV scheme as well.
Homomorphic operations that are based on key switching, i.e., relinearization and rotation, do not consume any noise budget (BFV) or impact accuracy (CKKS).
The implementations of these operations are significantly simplified and unified, and no longer use bit decomposition, so decomposition bit count is gone.
Moreover, fresh ciphertexts now have lower noise.
These changes have an effect on the API and it will be especially worthwhile for users of older versions of the library to study the examples and comments in [native/examples/3_levels.cpp](native/examples/3_levels.cpp) (C++) or [dotnet/examples/3_Levels.cs](dotnet/examples/3_Levels.cs) (C#).

The setup of `EncryptionParameters` has been made both easier and safer (see [API Changes](#api-changes) below).

The examples in [native/examples/](native/examples/) and [dotnet/examples/](dotnet/examples/) have been redesigned to better teach the multiple technical concepts required to use Microsoft SIGMA correctly and efficiently, and more compactly demonstrate the API.

### API Changes

Deleted header files:

- `native/defaultparameters.h`

New header files:

- [native/src/sigma/kswitchkeys.h](native/src/sigma/kswitchkeys.h): new base class for `RelinKeys` and `GaloisKeys`)
- [native/src/sigma/modulus.h](native/src/sigma/modulus.h): static helper functions for parameter selection
- [native/src/sigma/valcheck.h](native/src/sigma/valcheck.h): object validity check functionality
- [native/src/sigma/util/rlwe.h](native/src/sigma/util/rlwe.h)

In class `SIGMAContext`:

- Replaced `context_data(parms_id_type)` with `get_context_data(parms_id_type)`.
- Removed `context_data()`.
- Added `key_context_data()`, `key_parms_id()`, `first_context_data()`, and `last_context_data()`.
- Added `using_keyswitching()` that indicates whether key switching is supported in this `SIGMAContext`.
- `Create(...)` in C++, and constructor in C#, now accepts an optional security level based on [HomomorphicEncryption.org](https://HomomorphicEncryption.org) security standard, causing it to enforce the specified security level.
By default a 128-bit security level is used.
- Added `prev_context_data()` method to class `ContextData` (doubly linked modulus switching chain).
- In C# `SIGMAContext` now has a public constructor.

Parameter selection:

- Removed the `DefaultParams` class.
- Default `coeff_modulus` for the BFV scheme are now accessed through the function `CoeffModulus::BFVDefault(...)`.
These moduli are not recommended for the CKKS scheme.
- Customized `coeff_modulus` for the CKKS scheme can be created using `CoeffModulus::Create(...)` which takes the `poly_modulus_degree` and a vector of bit-lengths of the prime factors as arguments.
It samples suitable primes close to 2^bit_length and returns a vector of `SmallModulus` elements.
- `PlainModulus::Batching(...)` can be used to sample a prime for `plain_modulus` that supports `BatchEncoder` for the BFV scheme.

Other important changes:

- Removed `size_capacity` function and data members from `Ciphertext` class.
- Moved all validation methods such as `is_valid_for` and `is_metadata_valid_for` to `valcheck.h`.
- Removed argument `decomposition_bit_count` from methods `relin_keys(...)` and `galois_keys(...)` in class `KeyGenerator`.
- It is no longer possible to create more than one relinearization key.
This is to simplify the API and reduce confusion. We have never seen a real use-case where more relinearization keys would be a good idea.
- Added methods to generate an encryption of zero to `Encryptor`.
- Added comparison methods and primality check for `SmallModulus`.
- Classes `RelinKeys` and `GaloisKeys` are now derived from a common base class `KSwitchKeys`.
- GoogleTest framework is now included as a Git submodule.
- Numerous bugs have been fixed, particularly in the .NET wrappers.

## Version 3.2
