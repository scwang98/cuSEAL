
#include "kernelutils.cuh"
#include "util/randomgenerator.cuh"

namespace sigma::kernel_util {

    __global__ void g_dyadic_product_coeffmod(
            const uint64_t *operand1,
            const uint64_t *operand2,
            const uint64_t modulus_value,
            const uint64_t const_ratio_0,
            const uint64_t const_ratio_1,
            uint64_t *result) {

        auto tid = blockDim.x * blockIdx.x + threadIdx.x;

        // Reduces z using base 2^64 Barrett reduction
        uint64_t z[2], tmp1, tmp2[2], tmp3, carry;
        d_multiply_uint64(*(operand1 + tid), *(operand2 + tid), z);

        // Multiply input and const_ratio
        // Round 1
        d_multiply_uint64_hw64(z[0], const_ratio_0, &carry);
        d_multiply_uint64(z[0], const_ratio_1, tmp2);
        tmp3 = tmp2[1] + d_add_uint64(tmp2[0], carry, &tmp1);

        // Round 2
        d_multiply_uint64(z[1], const_ratio_0, tmp2);
        carry = tmp2[1] + d_add_uint64(tmp1, tmp2[0], &tmp1);

        // This is all we care about
        tmp1 = z[1] * const_ratio_1 + tmp3 + carry;

        // Barrett subtraction
        tmp3 = z[0] - tmp1 * modulus_value;

        // Claim: One more subtraction is enough
        *(result + tid) = tmp3 >= modulus_value ? tmp3 - modulus_value : tmp3;

    }

    void dyadic_product_coeffmod_inplace(
            uint64_t *operand1, const uint64_t *operand2,
            size_t coeff_count, size_t ntt_size, size_t coeff_modulus_size, const Modulus &modulus) {

        const uint64_t modulus_value = modulus.value();
        const uint64_t const_ratio_0 = modulus.const_ratio()[0];
        const uint64_t const_ratio_1 = modulus.const_ratio()[1];

        uint blockDim = coeff_count * ntt_size * coeff_modulus_size / 128;

        g_dyadic_product_coeffmod<<<blockDim, 128>>>(
                operand1,
                operand2,
                modulus_value,
                const_ratio_0,
                const_ratio_1,
                operand1);

    }

    void dyadic_product_coeffmod(
            const uint64_t *operand1, const uint64_t *operand2, size_t coeff_count, size_t ntt_size,
            size_t coeff_modulus_size, const Modulus &modulus, uint64_t *result, cudaStream_t &stream) {

        const uint64_t modulus_value = modulus.value();
        const uint64_t const_ratio_0 = modulus.const_ratio()[0];
        const uint64_t const_ratio_1 = modulus.const_ratio()[1];

        uint threadDim = 128;
        uint blockDim = coeff_count * ntt_size * coeff_modulus_size / threadDim;

        g_dyadic_product_coeffmod<<<blockDim, threadDim, 0, stream>>>(
                operand1,
                operand2,
                modulus_value,
                const_ratio_0,
                const_ratio_1,
                result);

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

        extern __shared__ uint64_t shared_array[];

#pragma unroll
        for (uint iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++) {
            auto global_tid = local_tid + iteration_num * 1024;
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
                ct_ntt_inner_single<8, 32768><<<8, 1024, 4096 * sizeof(uint64_t)>>>(operand, tables);
                break;
            }
            case 16384: {
                ct_ntt_inner<1, 16384><<<16384 / 1024 / 2, 1024>>>(operand, tables);
                ct_ntt_inner<2, 16384><<<16384 / 1024 / 2, 1024>>>(operand, tables);
                ct_ntt_inner_single<4, 16384><<<4, 1024, 4096 * sizeof(uint64_t)>>>(operand, tables);
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

    void g_ntt_negacyclic_harvey(uint64_t *operand, size_t coeff_count, const util::NTTTables &tables, cudaStream_t &stream) {
        switch (coeff_count) {
            case 32768: {
                ct_ntt_inner<1, 32768><<<32768 / 1024 / 2, 1024, 0, stream>>>(operand, tables);
                ct_ntt_inner<2, 32768><<<32768 / 1024 / 2, 1024, 0, stream>>>(operand, tables);
                ct_ntt_inner<4, 32768><<<32768 / 1024 / 2, 1024, 0, stream>>>(operand, tables);
                ct_ntt_inner_single<8, 32768><<<8, 1024, 4096 * sizeof(uint64_t), stream>>>(operand, tables);
                break;
            }
            case 16384: {
                ct_ntt_inner<1, 16384><<<16384 / 1024 / 2, 1024, 0, stream>>>(operand, tables);
                ct_ntt_inner<2, 16384><<<16384 / 1024 / 2, 1024, 0, stream>>>(operand, tables);
                ct_ntt_inner_single<4, 16384><<<4, 1024, 4096 * sizeof(uint64_t), stream>>>(operand, tables);
                break;
            }
            case 8192: {
                ct_ntt_inner<1, 8192><<<8192 / 1024 / 2, 1024, 0, stream>>>(operand, tables);
                ct_ntt_inner_single<2, 8192><<<2, 1024, 4096 * sizeof(uint64_t), stream>>>(operand, tables);
                break;
            }
            case 4096: {
                ct_ntt_inner_single<1, 4096> <<<1, 1024, 4096 * sizeof(uint64_t), stream>>>(operand, tables);
                break;
            }
            case 2048: {
                ct_ntt_inner_single<1, 2048> <<<1, 1024, 2048 * sizeof(uint64_t), stream>>>(operand, tables);
                break;
            }
            default:
                throw std::invalid_argument("not support");
        }
        CHECK(cudaGetLastError());
    }

    __device__ inline constexpr int d_hamming_weight(unsigned char value) {
        int t = static_cast<int>(value);
        t -= (t >> 1) & 0x55;
        t = (t & 0x33) + ((t >> 2) & 0x33);
        return (t + (t >> 4)) & 0x0F;
    }

    __global__
    void g_sample_poly_cbd(const Modulus *coeff_modulus, size_t coeff_modulus_size, size_t coeff_count, uint64_t *destination) {

        auto tid = blockDim.x * blockIdx.x + threadIdx.x;

        auto ptr = destination + tid;
        auto c_ptr = reinterpret_cast<unsigned char *>(ptr);
        c_ptr[2] &= 0x1F;
        c_ptr[5] &= 0x1F;
        int32_t noise = d_hamming_weight(c_ptr[0]) + d_hamming_weight(c_ptr[1]) + d_hamming_weight(c_ptr[2]) -
                d_hamming_weight(c_ptr[3]) - d_hamming_weight(c_ptr[4]) - d_hamming_weight(c_ptr[5]);
        auto flag = static_cast<uint64_t>(-static_cast<int64_t>(noise < 0));
        for (uint i = 0; i < coeff_modulus_size; ++i) {
            *(ptr + i * coeff_count) = static_cast<uint64_t>(noise) + (flag & (*(coeff_modulus + i)).value());
        }
    }

    void sample_poly_cbd(
            util::RandomGenerator *random_generator, const Modulus *coeff_modulus, size_t coeff_modulus_size,
            size_t coeff_count, uint64_t *destination, cudaStream_t &stream) {

        random_generator->generate(destination, coeff_count, stream);

        g_sample_poly_cbd<<<coeff_count / 1024, 1024, 0, stream>>>(coeff_modulus, coeff_modulus_size, coeff_count, destination);

    }

    __global__
    void g_add_negate_poly_coeffmod(
            const uint64_t *operand1, const uint64_t *operand2, const uint64_t *operand3, const uint64_t modulus_value,
            uint64_t *result) {
        auto tid = blockDim.x * blockIdx.x + threadIdx.x;

        std::uint64_t sum = operand1[tid] + operand2[tid];
        auto coeff = SIGMA_COND_SELECT(sum >= modulus_value, sum - modulus_value, sum);
        std::int64_t non_zero = (coeff != 0);
        coeff = (modulus_value - coeff) & static_cast<std::uint64_t>(-non_zero);
        sum = coeff + operand3[tid];
        result[tid] = SIGMA_COND_SELECT(sum >= modulus_value, sum - modulus_value, sum);
    }

    void add_negate_add_poly_coeffmod(
            const uint64_t *operand1, const uint64_t *operand2, const uint64_t *operand3, std::size_t coeff_count, uint64_t modulus_value,
            uint64_t *result, cudaStream_t &stream) {

        g_add_negate_poly_coeffmod<<<coeff_count / 1024, 1024, 0, stream>>>(operand1, operand2, operand3, modulus_value, result);

    }

    __global__
    void g_add_poly_coeffmod(
            const uint64_t *operand1, const uint64_t *operand2, const uint64_t modulus_value, uint64_t *result) {
        auto tid = blockDim.x * blockIdx.x + threadIdx.x;

        auto sum = operand1[tid] + operand2[tid];
        result[tid] = SIGMA_COND_SELECT(sum >= modulus_value, sum - modulus_value, sum);
    }

    void add_poly_coeffmod(
            const uint64_t *operand1, const uint64_t *operand2, size_t size, size_t coeff_modulus_size,
            std::size_t coeff_count, uint64_t modulus_value, uint64_t *result) {
        auto total_size = size * coeff_modulus_size * coeff_count;
        g_add_poly_coeffmod<<<total_size / 128, 128>>>(operand1, operand2, modulus_value, result);
    }

}
