
#pragma once

#include "kernelprovider.h"
#include "util/devicearray.cuh"
#include "util/hostarray.h"
#include "util/uintarithsmallmod.h"
#include "util/ntt.h"
#include "modulus.h"
#include "util/uint128_ntt.h"

#define KERNEL_CALL(funcname, n) size_t block_count = kernel_util::ceilDiv_(n, 256); funcname<<<block_count, 256>>>
#define GET_INDEX_COND_RETURN(n) size_t gindex = blockDim.x * blockIdx.x + threadIdx.x; if (gindex >= (n)) return
#define FOR_N(name, count) for (size_t name = 0; name < count; name++)

namespace sigma {

    namespace util {
        class RandomGenerator;
    }

    // static class. use as namespace
    class DeviceHelper {
    public:
        __device__ inline static uint64_t getModulusValue(const Modulus &m) {
            return m.value_;
        }

        __device__ inline static const uint64_t *getModulusConstRatio(const Modulus &m) {
            return static_cast<const uint64_t *>(&m.const_ratio_[0]);
        }
    };

    namespace kernel_util {

        using sigma::util::ConstDevicePointer;
        using sigma::util::DevicePointer;
        using CPointer = ConstDevicePointer<uint64_t>;
        using MPointer = ConstDevicePointer<Modulus>;
        using sigma::util::MultiplyUIntModOperand;
        using uint128_t = unsigned __int128;

        inline util::DeviceArray<uint64_t> kAllocate(uint64_t s) {
            return util::DeviceArray<uint64_t>(s);
        }

        inline util::DeviceArray<uint64_t> kAllocate(uint64_t s, uint64_t t) {
            return util::DeviceArray<uint64_t>(s * t);
        }

        inline util::DeviceArray<uint64_t> kAllocate(uint64_t s, uint64_t t, uint64_t u) {
            return util::DeviceArray<uint64_t>(s * t * u);
        }

        template <typename T>
        inline util::DeviceArray<T> kAllocateZero(size_t size) {
            auto ret = util::DeviceArray<T>(size);
            KernelProvider::memsetZero(ret.get(), ret.size());
            return ret;
        }

        inline util::DeviceArray<uint64_t> kAllocateZero(uint64_t s, uint64_t t) {
            auto ret = util::DeviceArray<uint64_t>(s * t);
            KernelProvider::memsetZero(ret.get(), ret.size());
            return ret;
        }

        inline util::DeviceArray<uint64_t> kAllocateZero(uint64_t s, uint64_t t, uint64_t u) {
            auto ret = util::DeviceArray<uint64_t>(s * t * u);
            KernelProvider::memsetZero(ret.get(), ret.size());
            return ret;
        }


        inline size_t ceilDiv_(size_t a, size_t b) {
            return (a % b) ? (a / b + 1) : (a / b);
        }

        __device__ inline unsigned char dAddUint64(uint64_t operand1, uint64_t operand2, uint64_t *result) {
            *result = operand1 + operand2;
            return static_cast<unsigned char>(*result < operand1);
        }

        __device__ inline void dMultiplyUint64HW64(uint64_t operand1, uint64_t operand2, uint64_t *hw64) {
            *hw64 = static_cast<uint64_t>(
                    ((static_cast<uint128_t>(operand1) * static_cast<uint128_t>(operand2)) >> 64));
        }

        __device__ inline void d_multiply_uint64_hw64(uint64_t operand1, uint64_t operand2, uint64_t *hw64) {
            *hw64 = static_cast<uint64_t>(
                    ((static_cast<uint128_t>(operand1) * static_cast<uint128_t>(operand2)) >> 64));
        }

        __device__ inline void d_multiply_uint64(uint64_t operand1, uint64_t operand2, uint64_t *result128) {
            uint128_t product = static_cast<uint128_t>(operand1) * operand2;
            result128[0] = static_cast<uint64_t>(product);
            result128[1] = static_cast<uint64_t>(product >> 64);
        }

        __device__ inline unsigned char d_add_uint64(uint64_t operand1, uint64_t operand2, uint64_t *result) {
            *result = operand1 + operand2;
            return static_cast<unsigned char>(*result < operand1);
        }

        __device__ inline uint64_t dBarrettReduce64(uint64_t input, const Modulus &modulus) {
            uint64_t tmp[2];
            const std::uint64_t *const_ratio = DeviceHelper::getModulusConstRatio(modulus);
            dMultiplyUint64HW64(input, const_ratio[1], tmp + 1);
            uint64_t modulusValue = DeviceHelper::getModulusValue(modulus);

            // Barrett subtraction
            tmp[0] = input - tmp[1] * modulusValue;

            // One more subtraction is enough
            return (tmp[0] >= modulusValue) ? (tmp[0] - modulusValue) : (tmp[0]);
        }

        __device__ inline uint64_t d_barrett_reduce_64(uint64_t input, const Modulus &modulus) {
            uint64_t tmp[2];
            const std::uint64_t *const_ratio = modulus.const_ratio();
            d_multiply_uint64_hw64(input, const_ratio[1], tmp + 1);
            uint64_t modulusValue = modulus.value();

            // Barrett subtraction
            tmp[0] = input - tmp[1] * modulusValue;

            // One more subtraction is enough
            return (tmp[0] >= modulusValue) ? (tmp[0] - modulusValue) : (tmp[0]);
        }

        __device__
        inline uint64_t d_multiply_uint_mod_lazy(std::uint64_t x, MultiplyUIntModOperand y, const Modulus &modulus) {
            uint64_t tmp1;
            const uint64_t p = modulus.value();
            d_multiply_uint64_hw64(x, y.quotient, &tmp1);
            return y.operand * x - tmp1 * p;
        }

        __device__
        inline void dMultiplyUint64(uint64_t operand1, uint64_t operand2, uint64_t *result128) {
            uint128_t product = static_cast<uint128_t>(operand1) * operand2;
            result128[0] = static_cast<unsigned long long>(product);
            result128[1] = static_cast<unsigned long long>(product >> 64);
        }

        __device__ inline uint64_t dBarrettReduce128(const uint64_t *input, const Modulus &modulus) {
            // Reduces input using base 2^64 Barrett reduction
            // input allocation size must be 128 bits

            uint64_t tmp1, tmp2[2], tmp3, carry;
            const std::uint64_t *const_ratio = DeviceHelper::getModulusConstRatio(modulus);

            // Multiply input and const_ratio
            // Round 1
            d_multiply_uint64_hw64(input[0], const_ratio[0], &carry);

            dMultiplyUint64(input[0], const_ratio[1], tmp2);
            tmp3 = tmp2[1] + dAddUint64(tmp2[0], carry, &tmp1);

            // Round 2
            dMultiplyUint64(input[1], const_ratio[0], tmp2);
            carry = tmp2[1] + dAddUint64(tmp1, tmp2[0], &tmp1);

            // This is all we care about
            tmp1 = input[1] * const_ratio[1] + tmp3 + carry;

            // Barrett subtraction
            uint64_t modulus_value = DeviceHelper::getModulusValue(modulus);
            tmp3 = input[0] - tmp1 * modulus_value;

            // One more subtraction is enough
            return (tmp3 >= modulus_value) ? (tmp3 - modulus_value) : (tmp3);
        }

        __device__ inline std::uint64_t dNegateUintMod(std::uint64_t operand, const Modulus &modulus) {
            std::int64_t non_zero = static_cast<std::int64_t>(operand != 0);
            return (DeviceHelper::getModulusValue(modulus) - operand) & static_cast<std::uint64_t>(-non_zero);
        }

        void g_ntt_negacyclic_harvey(uint64_t *operand, size_t coeff_count, const util::NTTTables &tables);

        void dyadic_product_coeffmod_inplace(
                uint64_t *operand1, const uint64_t *operand2,
                size_t coeff_count, size_t ntt_size, size_t coeff_modulus_size, const Modulus &modulus);

        void dyadic_product_coeffmod(
                const uint64_t *operand1, const uint64_t *operand2, size_t coeff_count,size_t ntt_size,
                size_t coeff_modulus_size, const Modulus &modulus, uint64_t *result);

        void sample_poly_cbd(util::RandomGenerator *random_generator, const Modulus *coeff_modulus, size_t coeff_modulus_size, size_t coeff_count, uint64_t *destination);

        void add_negate_add_poly_coeffmod(
                const uint64_t *operand1, const uint64_t *operand2, const uint64_t *operand3, std::size_t coeff_count, uint64_t modulus_value,
                uint64_t *result);



    }
}
