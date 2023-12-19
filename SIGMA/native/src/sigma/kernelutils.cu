#include "kernelutils.cuh"
#include "assert.h"

#define KERNEL_CALL(funcname, n) size_t block_count = ceilDiv_(n, 256); funcname<<<block_count, 256>>>
#define POLY_ARRAY_ARGUMENTS size_t poly_size, size_t coeff_modulus_size, size_t poly_modulus_degree
#define POLY_ARRAY_ARGCALL poly_size, coeff_modulus_size, poly_modulus_degree
#define GET_INDEX size_t gindex = blockDim.x * blockIdx.x + threadIdx.x
#define GET_INDEX_COND_RETURN(n) size_t gindex = blockDim.x * blockIdx.x + threadIdx.x; if (gindex >= (n)) return
#define FOR_N(name, count) for (size_t name = 0; name < count; name++)


namespace sigma::kernel_util {

    __global__ void gAddPolyCoeffmod(
            const uint64_t *operand1,
            const uint64_t *operand2,
            POLY_ARRAY_ARGUMENTS,
            const Modulus *modulus,
            uint64_t *result
    ) {
        GET_INDEX_COND_RETURN(poly_modulus_degree);
        FOR_N(rns_index, coeff_modulus_size) {
            const uint64_t modulusValue = DeviceHelper::getModulusValue(modulus[rns_index]);
            FOR_N(poly_index, poly_size) {
                size_t id = (poly_index * coeff_modulus_size + rns_index) * poly_modulus_degree + gindex;
                uint64_t sum = operand1[id] + operand2[id];
                result[id] = sum >= modulusValue ? sum - modulusValue : sum;
            }
        }
    }

    void kAddPolyCoeffmod(
            CPointer operand1,
            CPointer operand2,
            POLY_ARRAY_ARGUMENTS,
            MPointer modulus,
            DevicePointer<uint64_t> result
    ) {
        KERNEL_CALL(gAddPolyCoeffmod, poly_modulus_degree)(
                operand1.get(), operand2.get(),
                POLY_ARRAY_ARGCALL, modulus.get(), result.get()
        );
    }

    __device__ inline uint64_t
    dDyadicSingle(uint64_t o1, uint64_t o2, uint64_t modulus_value, uint64_t const_ratio_0, uint64_t const_ratio_1) {

        uint64_t z[2], tmp1, tmp2[2], tmp3, carry;

        // Reduces z using base 2^64 Barrett reduction
        dMultiplyUint64(o1, o2, z);

        // Multiply input and const_ratio
        // Round 1
        d_multiply_uint64_hw64(z[0], const_ratio_0, &carry);
        dMultiplyUint64(z[0], const_ratio_1, tmp2);
        tmp3 = tmp2[1] + dAddUint64(tmp2[0], carry, &tmp1);

        // Round 2
        dMultiplyUint64(z[1], const_ratio_0, tmp2);
        carry = tmp2[1] + dAddUint64(tmp1, tmp2[0], &tmp1);

        // This is all we care about
        tmp1 = z[1] * const_ratio_1 + tmp3 + carry;

        // Barrett subtraction
        tmp3 = z[0] - tmp1 * modulus_value;

        // Claim: One more subtraction is enough
        uint64_t sum = ((tmp3 >= modulus_value) ? (tmp3 - modulus_value) : (tmp3));
        return sum;
    }

    __global__ void gDyadicConvolutionCoeffmod(
            const uint64_t *operand1,
            const uint64_t *operand2_reversed,
            POLY_ARRAY_ARGUMENTS,
            const Modulus *moduli,
            uint64_t *single_poly_result_accumulator
    ) {
        GET_INDEX_COND_RETURN(poly_modulus_degree);
        FOR_N(rns_index, coeff_modulus_size) {
            const uint64_t modulus_value = DeviceHelper::getModulusValue(moduli[rns_index]);
            const uint64_t const_ratio_0 = DeviceHelper::getModulusConstRatio(moduli[rns_index])[0];
            const uint64_t const_ratio_1 = DeviceHelper::getModulusConstRatio(moduli[rns_index])[1];
            FOR_N(poly_index, poly_size) {

                const uint64_t *o1 = operand1
                                     + (poly_index * coeff_modulus_size + rns_index) * poly_modulus_degree + gindex;
                const uint64_t *o2 = operand2_reversed - poly_index * coeff_modulus_size * poly_modulus_degree
                                     + rns_index * poly_modulus_degree + gindex;
                uint64_t *res = single_poly_result_accumulator
                                + rns_index * poly_modulus_degree + gindex;

                // Claim: One more subtraction is enough
                uint64_t sum = *res + dDyadicSingle(*o1, *o2, modulus_value, const_ratio_0, const_ratio_1);
                *res = sum >= modulus_value ? sum - modulus_value : sum;
            }
        }
    }

    void kDyadicConvolutionCoeffmod(
            CPointer operand1,
            CPointer operand2_reversed,
            POLY_ARRAY_ARGUMENTS,
            MPointer moduli,
            DevicePointer<uint64_t> single_poly_result_accumulator
    ) {
        KERNEL_CALL(gDyadicConvolutionCoeffmod, poly_modulus_degree)(
                operand1.get(),
                operand2_reversed.get(),
                POLY_ARRAY_ARGCALL,
                moduli.get(), single_poly_result_accumulator.get()
        );
    }

    __global__ void gDyadicProductCoeffmod(
            const uint64_t *operand1,
            const uint64_t *operand2,
            POLY_ARRAY_ARGUMENTS,
            const Modulus *moduli,
            uint64_t *output
    ) {
        GET_INDEX_COND_RETURN(poly_modulus_degree);
        FOR_N(rns_index, coeff_modulus_size) {
            const uint64_t modulus_value = DeviceHelper::getModulusValue(moduli[rns_index]);
            const uint64_t const_ratio_0 = DeviceHelper::getModulusConstRatio(moduli[rns_index])[0];
            const uint64_t const_ratio_1 = DeviceHelper::getModulusConstRatio(moduli[rns_index])[1];
            FOR_N(poly_index, poly_size) {
                size_t id = (poly_index * coeff_modulus_size + rns_index) * poly_modulus_degree + gindex;
                output[id] = dDyadicSingle(operand1[id], operand2[id], modulus_value, const_ratio_0, const_ratio_1);
            }
        }
    }

    void kDyadicProductCoeffmod(
            CPointer operand1,
            CPointer operand2,
            POLY_ARRAY_ARGUMENTS,
            MPointer moduli,
            DevicePointer<uint64_t> output
    ) {
        KERNEL_CALL(gDyadicProductCoeffmod, poly_modulus_degree)(
                operand1.get(),
                operand2.get(),
                POLY_ARRAY_ARGCALL,
                moduli.get(), output.get()
        );
    }

    __global__ void gDyadicSquareCoeffmod(
            const uint64_t *operand,
            size_t coeff_modulus_size,
            size_t poly_modulus_degree,
            const Modulus *moduli,
            uint64_t *output
    ) {
        GET_INDEX_COND_RETURN(poly_modulus_degree);
        size_t d = coeff_modulus_size * poly_modulus_degree;
        FOR_N(rns_index, coeff_modulus_size) {
            const uint64_t modulus_value = DeviceHelper::getModulusValue(moduli[rns_index]);
            const uint64_t const_ratio_0 = DeviceHelper::getModulusConstRatio(moduli[rns_index])[0];
            const uint64_t const_ratio_1 = DeviceHelper::getModulusConstRatio(moduli[rns_index])[1];
            size_t id = rns_index * poly_modulus_degree + gindex;
            output[2 * d + id] = dDyadicSingle(operand[1 * d + id], operand[1 * d + id], modulus_value, const_ratio_0,
                                               const_ratio_1);
            uint64_t cross = dDyadicSingle(operand[0 * d + id], operand[1 * d + id], modulus_value, const_ratio_0,
                                           const_ratio_1);
            cross += cross;
            output[1 * d + id] = cross >= modulus_value ? cross - modulus_value : cross;
            output[0 * d + id] = dDyadicSingle(operand[0 * d + id], operand[0 * d + id], modulus_value, const_ratio_0,
                                               const_ratio_1);
        }
    }

    void kDyadicSquareCoeffmod(
            CPointer operand,
            size_t coeff_modulus_size,
            size_t poly_modulus_degree,
            MPointer moduli,
            DevicePointer<uint64_t> output
    ) {
        KERNEL_CALL(gDyadicSquareCoeffmod, poly_modulus_degree)(
                operand.get(), coeff_modulus_size, poly_modulus_degree,
                moduli.get(), output.get()
        );
    }

    __global__ void gModBoundedUsingNttTables(
            uint64_t *operand,
            POLY_ARRAY_ARGUMENTS,
            const util::NTTTables *ntt_tables) {
        GET_INDEX_COND_RETURN(poly_modulus_degree);
        FOR_N(rns_index, coeff_modulus_size) {
            uint64_t modulus_value = DeviceHelper::getModulusValue(ntt_tables[rns_index].modulus());
            uint64_t twice_modulus_value = modulus_value << 1;
            FOR_N(poly_index, poly_size) {
                size_t id = (poly_index * coeff_modulus_size + rns_index) * poly_modulus_degree + gindex;
                if (operand[id] >= twice_modulus_value) operand[id] -= twice_modulus_value;
                if (operand[id] >= modulus_value) operand[id] -= modulus_value;
            }
        }
    }

    void kModBoundedUsingNttTables(
            uint64_t *operand,
            POLY_ARRAY_ARGUMENTS,
            ConstDevicePointer<util::NTTTables> ntt_tables) {
        KERNEL_CALL(gModBoundedUsingNttTables, poly_modulus_degree)(
                operand, POLY_ARRAY_ARGCALL, ntt_tables.get()
        );
    }

    __global__ void gModuloPolyCoeffs(
            const uint64_t *operand,
            POLY_ARRAY_ARGUMENTS,
            const Modulus *moduli,
            uint64_t *result
    ) {
        GET_INDEX_COND_RETURN(poly_modulus_degree);
        FOR_N(rns_index, coeff_modulus_size) {
            FOR_N(poly_index, poly_size) {
                size_t id = (poly_index * coeff_modulus_size + rns_index) * poly_modulus_degree + gindex;
                result[id] = d_barrett_reduce_64(operand[id], moduli[rns_index]);
            }
        }
    }

    void kModuloPolyCoeffs(
            CPointer operand,
            POLY_ARRAY_ARGUMENTS,
            MPointer moduli,
            DevicePointer<uint64_t> result
    ) {
        KERNEL_CALL(gModuloPolyCoeffs, poly_modulus_degree)(
                operand.get(),
                POLY_ARRAY_ARGCALL,
                moduli.get(), result.get()
        );
    }

    __global__ void gMultiplyInvDegreeNttTables(
            uint64_t *poly_array,
            POLY_ARRAY_ARGUMENTS,
            const util::NTTTables *ntt_tables
    ) {
        GET_INDEX_COND_RETURN(poly_modulus_degree);
        FOR_N(rns_index, coeff_modulus_size) {
            const Modulus &modulus = ntt_tables[rns_index].modulus();
            MultiplyUIntModOperand scalar = ntt_tables[rns_index].inv_degree_modulo();;
            FOR_N(poly_index, poly_size) {
                size_t id = (poly_index * coeff_modulus_size + rns_index) * poly_modulus_degree + gindex;
                poly_array[id] = dMultiplyUintModLazy(poly_array[id], scalar, modulus);
            }
        }
    }

    __global__ void gMultiplyPolyScalarCoeffmod(
            const uint64_t *poly_array,
            POLY_ARRAY_ARGUMENTS,
            const MultiplyUIntModOperand *reduced_scalar,
            const Modulus *modulus,
            uint64_t *result) {
        GET_INDEX_COND_RETURN(poly_modulus_degree);
        FOR_N(rns_index, coeff_modulus_size) {
            FOR_N(poly_index, poly_size) {
                size_t id = (poly_index * coeff_modulus_size + rns_index) * poly_modulus_degree + gindex;
                result[id] = dMultiplyUintMod(poly_array[id], reduced_scalar[rns_index], modulus[rns_index]);
            }
        }
    }


    void kMultiplyPolyScalarCoeffmod(CPointer poly_array, POLY_ARRAY_ARGUMENTS, uint64_t scalar, MPointer modulus,
                                     DevicePointer<uint64_t> result) {
        util::DeviceArray<MultiplyUIntModOperand> reduced_scalar(coeff_modulus_size);
        assert(coeff_modulus_size <= 256);
        gSetMultiplyUIntModOperand<<<1, coeff_modulus_size>>>(scalar, modulus.get(), coeff_modulus_size,
                                                              reduced_scalar.get());
        KERNEL_CALL(gMultiplyPolyScalarCoeffmod, poly_modulus_degree)(
                poly_array.get(), POLY_ARRAY_ARGCALL, reduced_scalar.get(),
                modulus.get(), result.get()
        );
    }

    __global__ void gNegatePolyCoeffmod(
            const uint64_t *poly_array,
            POLY_ARRAY_ARGUMENTS,
            const Modulus *modulus,
            uint64_t *result
    ) {
        GET_INDEX_COND_RETURN(poly_modulus_degree);
        FOR_N(rns_index, coeff_modulus_size) {
            auto modulus_value = DeviceHelper::getModulusValue(modulus[rns_index]);
            FOR_N(poly_index, poly_size) {
                size_t id = (poly_index * coeff_modulus_size + rns_index) * poly_modulus_degree + gindex;
                uint64_t coeff = poly_array[id];
                int64_t non_zero = (coeff != 0);
                result[id] = (modulus_value - coeff) & static_cast<uint64_t>(-non_zero);
            }
        }
    }

    void kNttTransferFromRev(
            DevicePointer<uint64_t> operand,
            size_t poly_size,
            size_t coeff_modulus_size,
            size_t poly_modulus_degree_power,
            ConstDevicePointer<util::NTTTables> ntt_tables,
            bool use_inv_root_powers
    ) {
        std::size_t n = size_t(1) << poly_modulus_degree_power;
        std::size_t m = n >> 1;
        std::size_t layer = 0;
        for (; m >= 1; m >>= 1) {
            kNttTransferFromRevLayered(
                    layer, operand,
                    poly_size, coeff_modulus_size,
                    poly_modulus_degree_power, ntt_tables,
                    use_inv_root_powers);
            layer++;
        }
        KERNEL_CALL(gMultiplyInvDegreeNttTables, n)(
                operand.get(), poly_size, coeff_modulus_size, n, ntt_tables.get()
        );
    }

    __global__ void gNttTransferFromRevLayered(
            size_t layer,
            uint64_t *operand,
            size_t poly_size,
            size_t coeff_modulus_size,
            size_t poly_modulus_degree_power,
            const util::NTTTables *ntt_tables,
            bool use_inv_root_powers
    ) {
        GET_INDEX_COND_RETURN(1 << (poly_modulus_degree_power - 1));
        size_t m = 1 << (poly_modulus_degree_power - 1 - layer);
        size_t gap_power = layer;
        size_t gap = 1 << gap_power;
        size_t rid = (1 << poly_modulus_degree_power) - (m << 1) + 1 + (gindex >> gap_power);
        size_t coeff_index = ((gindex >> gap_power) << (gap_power + 1)) + (gindex & (gap - 1));
        // printf("m = %lu, coeff_index = %lu\n", m, coeff_index);
        uint64_t u, v;
        FOR_N(rns_index, coeff_modulus_size) {
            const Modulus &modulus = ntt_tables[rns_index].modulus();
            uint64_t two_times_modulus = DeviceHelper::getModulusValue(modulus) << 1;
            MultiplyUIntModOperand r = use_inv_root_powers ?
                                       (ntt_tables[rns_index].get_from_device_inv_root_powers()[rid]) :
                                       (ntt_tables[rns_index].get_from_device_root_powers()[rid]);
            FOR_N(poly_index, poly_size) {
                uint64_t *x = operand + ((poly_index * coeff_modulus_size + rns_index) << poly_modulus_degree_power) +
                              coeff_index;
                uint64_t *y = x + gap;
                // printf("m=%lu,dx=%lu,u=%llu,v=%llu,r=(%llu,%llu),dr=%lu\n", m,
                //     ((poly_index * coeff_modulus_size + rns_index) << poly_modulus_degree_power) + coeff_index,
                //     *x, *y, r.operand, r.quotient, rid);
                u = *x;
                v = *y;
                *x = (u + v > two_times_modulus) ? (u + v - two_times_modulus) : (u + v);
                *y = dMultiplyUintModLazy(u + two_times_modulus - v, r, modulus);
                // printf("m = %lu xid = %lu u = %llu v = %llu x = %llu y = %llu\n", m, ((poly_index * coeff_modulus_size + rns_index) << poly_modulus_degree_power) + coeff_index, u, v, *x, *y);
            }
        }
    }

    void kNttTransferFromRevLayered(
            size_t layer,
            DevicePointer<uint64_t> operand,
            size_t poly_size,
            size_t coeff_modulus_size,
            size_t poly_modulus_degree_power,
            ConstDevicePointer<util::NTTTables> ntt_tables,
            bool use_inv_root_powers
    ) {
        std::size_t n = size_t(1) << poly_modulus_degree_power;
        KERNEL_CALL(gNttTransferFromRevLayered, n)(
                layer, operand.get(), poly_size, coeff_modulus_size,
                poly_modulus_degree_power, ntt_tables.get(),
                use_inv_root_powers
        );
    }

    __global__ void
    gSetMultiplyUIntModOperand(uint64_t scalar, const Modulus *moduli, size_t n, MultiplyUIntModOperand *result) {
        GET_INDEX_COND_RETURN(n);
        uint64_t reduced = d_barrett_reduce_64(scalar, moduli[gindex]);
        result[gindex].operand = reduced;
        std::uint64_t wide_quotient[2]{0, 0};
        std::uint64_t wide_coeff[2]{0, result[gindex].operand};
        dDivideUint128Inplace(wide_coeff, DeviceHelper::getModulusValue(moduli[gindex]), wide_quotient);
        result[gindex].quotient = wide_quotient[0];
    }

    __global__ void gSubPolyCoeffmod(
            const uint64_t *operand1,
            const uint64_t *operand2,
            POLY_ARRAY_ARGUMENTS,
            const Modulus *modulus,
            uint64_t *result
    ) {
        GET_INDEX_COND_RETURN(poly_modulus_degree);
        FOR_N(rns_index, coeff_modulus_size) {
            const uint64_t modulusValue = DeviceHelper::getModulusValue(modulus[rns_index]);
            FOR_N(poly_index, poly_size) {
                size_t id = (poly_index * coeff_modulus_size + rns_index) * poly_modulus_degree + gindex;
                uint64_t temp_result;
                int64_t borrow = dSubUint64(operand1[id], operand2[id], &temp_result);
                result[id] = temp_result + (modulusValue & static_cast<std::uint64_t>(-borrow));
            }
        }
    }

    void kSubPolyCoeffmod(
            CPointer operand1,
            CPointer operand2,
            POLY_ARRAY_ARGUMENTS,
            MPointer modulus,
            DevicePointer<uint64_t> result
    ) {
        KERNEL_CALL(gSubPolyCoeffmod, poly_modulus_degree)(
                operand1.get(), operand2.get(),
                POLY_ARRAY_ARGCALL, modulus.get(), result.get()
        );
    }

    __global__ void gNegacyclicShiftPolyCoeffmod(
            const uint64_t *poly,
            POLY_ARRAY_ARGUMENTS,
            size_t shift,
            const Modulus *modulus,
            uint64_t *result
    ) {
        GET_INDEX_COND_RETURN(poly_modulus_degree);
        uint64_t index_raw = shift + gindex;
        uint64_t index = index_raw & (static_cast<uint64_t>(poly_modulus_degree) - 1);
        FOR_N(rns_index, coeff_modulus_size) {
            const uint64_t modulusValue = DeviceHelper::getModulusValue(modulus[rns_index]);
            FOR_N(poly_index, poly_size) {
                size_t id = (poly_index * coeff_modulus_size + rns_index) * poly_modulus_degree + gindex;
                size_t rid = (poly_index * coeff_modulus_size + rns_index) * poly_modulus_degree + index;
                if (shift == 0) {
                    result[id] = poly[id];
                } else {
                    if (!(index_raw & static_cast<uint64_t>(poly_modulus_degree)) || !poly[id]) {
                        result[rid] = poly[id];
                    } else {
                        result[rid] = modulusValue - poly[id];
                    }
                }
            }
        }
    }


    void kNegacyclicShiftPolyCoeffmod(
            CPointer poly,
            POLY_ARRAY_ARGUMENTS,
            size_t shift,
            MPointer modulus,
            DevicePointer<uint64_t> result
    ) {
        KERNEL_CALL(gNegacyclicShiftPolyCoeffmod, poly_modulus_degree)(
                poly.get(), POLY_ARRAY_ARGCALL, shift, modulus.get(), result.get()
        );
    }

    __global__ void gMultiplyInvPolyDegreeCoeffmod(
            const uint64_t *poly_array,
            POLY_ARRAY_ARGUMENTS,
            const util::NTTTables *ntt_tables,
            const Modulus *modulus,
            uint64_t *result) {
        GET_INDEX_COND_RETURN(poly_modulus_degree);
        FOR_N(rns_index, coeff_modulus_size) {
            FOR_N(poly_index, poly_size) {
                size_t id = (poly_index * coeff_modulus_size + rns_index) * poly_modulus_degree + gindex;
                result[id] = dMultiplyUintMod(poly_array[id], ntt_tables[rns_index].inv_degree_modulo(),
                                              modulus[rns_index]);
            }
        }
    }

    void kMultiplyInvPolyDegreeCoeffmod(CPointer poly_array, POLY_ARRAY_ARGUMENTS,
                                        ConstDevicePointer<util::NTTTables> ntt_tables,
                                        MPointer modulus, DevicePointer<uint64_t> result) {
        assert(coeff_modulus_size <= 256);
        KERNEL_CALL(gMultiplyInvPolyDegreeCoeffmod, poly_modulus_degree)(
                poly_array.get(), POLY_ARRAY_ARGCALL, ntt_tables.get(),
                modulus.get(), result.get()
        );
    }

    template<unsigned l, unsigned n>
    __global__ void ct_ntt_inner(uint64_t *values, const util::NTTTables &tables) {

        const MultiplyUIntModOperand *roots = tables.get_from_device_root_powers();
        const Modulus &modulus = tables.modulus();

        auto modulus_value = modulus.value();
        auto two_times_modulus = modulus_value << 1;

        auto global_tid = blockIdx.x * 1024 + threadIdx.x;
        auto step = (n / l) / 2;
        auto psi_step = global_tid / step;
        auto target_index = psi_step * step * 2 + global_tid % step;

        const MultiplyUIntModOperand &r = roots[l + psi_step];

        uint64_t &x = values[target_index];
        uint64_t &y = values[target_index + step];
        uint64_t u = x >= two_times_modulus ? x - two_times_modulus : x;
        uint64_t v = d_multiply_uint_mod_lazy(y, r, modulus);
        x = u + v;
        y = u + two_times_modulus - v;
    }

    template<uint l, uint n>
    __global__ void ct_ntt_inner_single(uint64_t *values, const util::NTTTables &tables) {
        auto local_tid = threadIdx.x;

        const MultiplyUIntModOperand *roots = tables.get_from_device_root_powers();
        const Modulus &modulus = tables.modulus();

        extern __shared__ uint64_t shared_array[];  // declaration of shared_array

#pragma unroll
        for (uint iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++) {
            auto global_tid = local_tid + iteration_num * 1024;
            auto index = global_tid + blockIdx.x * (n / l);
            shared_array[global_tid] = values[global_tid + blockIdx.x * (n / l)];
        }

        auto modulus_value = modulus.value();
        auto two_times_modulus = modulus_value << 1;

        auto step = n / l;
#pragma unroll
        for (uint length = l; length < n; length <<= 1) {
            step >>= 1;

#pragma unroll
            for (uint iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++) {
                auto global_tid = local_tid + iteration_num * 1024;
                auto psi_step = global_tid / step;
                auto target_index = psi_step * step * 2 + global_tid % step;
                psi_step = (global_tid + blockIdx.x * (n / l / 2)) / step;

                const MultiplyUIntModOperand &r = roots[length + psi_step];

                uint64_t &x = shared_array[target_index];
                uint64_t &y = shared_array[target_index + step];
                uint64_t u = x >= two_times_modulus ? x - two_times_modulus : x;
                uint64_t v = d_multiply_uint_mod_lazy(y, r, modulus);
                x = u + v;
                y = u + two_times_modulus - v;
            }
            __syncthreads();
        }

        uint64_t value;
#pragma unroll
        for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++) {
            auto global_tid = local_tid + iteration_num * 1024;

            value = shared_array[global_tid];
            if (value >= two_times_modulus) {
                value -= two_times_modulus;
            }
            if (value >= modulus_value) {
                value -= modulus_value;
            }

            values[global_tid + blockIdx.x * (n / l)] = value;
        }
    }

    void g_ntt_negacyclic_harvey(uint64_t *operand, size_t coeff_count, const util::NTTTables &tables) {
        switch (coeff_count) {
            case 32768: {
                ct_ntt_inner<1, 32768><<<32768 / 1024 / 2, 1024>>>(operand, tables);
                ct_ntt_inner<2, 32768><<<32768 / 1024 / 2, 1024>>>(operand, tables);
                ct_ntt_inner<4, 32768><<<32768 / 1024 / 2, 1024>>>(operand, tables);
                ct_ntt_inner_single<2, 8192><<<2, 1024, 4096 * sizeof(uint64_t)>>>(operand, tables);
                break;
            }
            case 16384: {
                ct_ntt_inner<1, 16384><<<16384 / 1024 / 2, 1024>>>(operand, tables);
                ct_ntt_inner<2, 16384><<<16384 / 1024 / 2, 1024>>>(operand, tables);
                ct_ntt_inner_single<2, 8192><<<2, 1024, 4096 * sizeof(uint64_t)>>>(operand, tables);
                break;
            }
            case 8192: {
                ct_ntt_inner<1, 8192><<<8192 / 1024 / 2, 1024>>>(operand, tables);
                ct_ntt_inner_single<2, 8192><<<2, 1024, 4096 * sizeof(uint64_t)>>>(operand, tables);
                break;
            }
            case 4096: {
                ct_ntt_inner_single<1, 4096> <<<1, 1024, 4096 * sizeof(uint64_t)>>>(operand, tables);
                break;
            }
            case 2048: {
                ct_ntt_inner_single<1, 2048> <<<1, 1024, 2048 * sizeof(uint64_t)>>>(operand, tables);
                break;
            }
            default:
                throw std::invalid_argument("not support");
        }
        CHECK(cudaGetLastError());
    }

}
