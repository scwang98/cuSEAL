
#pragma once

#include "kernelprovider.cuh"
#include "util/devicearray.cuh"
#include "util/hostarray.h"
#include "util/uintarithsmallmod.h"
#include "util/ntt.h"
#include "modulus.h"
#include "util/uint128_ntt.h"

namespace sigma {

    namespace util {
        class RandomGenerator;
    }

    namespace kernel_util {

        using sigma::util::MultiplyUIntModOperand;

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

        __device__ inline std::uint64_t d_negate_uint_mod(std::uint64_t operand, const Modulus &modulus) {
            auto non_zero = static_cast<std::int64_t>(operand != 0);
            return (modulus.value() - operand) & static_cast<std::uint64_t>(-non_zero);
        }

        void g_ntt_negacyclic_harvey(uint64_t *operand, size_t coeff_count, const util::NTTTables &tables);

        void g_ntt_negacyclic_harvey(uint64_t *operand, size_t coeff_count, const util::NTTTables &tables, cudaStream_t &cudaStream);

        void dyadic_product_coeffmod(
                const uint64_t *operand1, const uint64_t *operand2, size_t coeff_count, size_t ntt_size,
                size_t coeff_modulus_size, const Modulus &modulus, uint64_t *result);

        void dyadic_product_coeffmod(
                const uint64_t *operand1, const uint64_t *operand2, size_t coeff_count,size_t ntt_size,
                size_t coeff_modulus_size, const Modulus &modulus, uint64_t *result, cudaStream_t &stream);

        void sample_poly_cbd(
                util::RandomGenerator *random_generator, const Modulus *coeff_modulus, size_t coeff_modulus_size,
                size_t coeff_count, uint64_t *destination);

        void sample_poly_cbd(
                util::RandomGenerator *random_generator, const Modulus *coeff_modulus, size_t coeff_modulus_size,
                size_t coeff_count, uint64_t *destination, cudaStream_t &stream);

        void add_negate_add_poly_coeffmod(
                const uint64_t *operand1, const uint64_t *operand2, const uint64_t *operand3, std::size_t coeff_count,
                uint64_t modulus_value, uint64_t *result);

        void add_negate_add_poly_coeffmod(
                const uint64_t *operand1, const uint64_t *operand2, const uint64_t *operand3, std::size_t coeff_count,
                uint64_t modulus_value, uint64_t *result, cudaStream_t &stream);

        void add_poly_coeffmod(
                const uint64_t *operand1, const uint64_t *operand2, size_t size, size_t coeff_modulus_size,
                std::size_t coeff_count, uint64_t modulus_value, uint64_t *result);

    }
}
