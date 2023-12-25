// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "sigma/ckks.h"
#include "kernelutils.cuh"
#include "cuComplex.h"
#include <random>
#include <stdexcept>
#include <cfloat>

using namespace std;
using namespace sigma::util;

namespace sigma
{
    CKKSEncoder::CKKSEncoder(const SIGMAContext &context) : context_(context)
    {
        // Verify parameters
        if (!context_.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }

        auto &context_data = *context_.first_context_data();
        if (context_data.parms().scheme() != scheme_type::ckks)
        {
            throw invalid_argument("unsupported scheme");
        }

        size_t coeff_count = context_data.parms().poly_modulus_degree();
        slots_ = coeff_count >> 1;
        int logn = get_power_of_two(coeff_count);

        host_matrix_reps_index_map_ = HostArray<size_t>(coeff_count);

        // Copy from the matrix to the value vectors
        uint64_t gen = 3;
        uint64_t pos = 1;
        uint64_t m = static_cast<uint64_t>(coeff_count) << 1;
        for (size_t i = 0; i < slots_; i++)
        {
            // Position in normal bit order
            uint64_t index1 = (pos - 1) >> 1;
            uint64_t index2 = (m - pos - 1) >> 1;

            // Set the bit-reversed locations
            host_matrix_reps_index_map_[i] = safe_cast<size_t>(reverse_bits(index1, logn));
            host_matrix_reps_index_map_[slots_ | i] = safe_cast<size_t>(reverse_bits(index2, logn));

            // Next primitive root
            pos *= gen;
            pos &= (m - 1);
        }

        matrix_reps_index_map_ = host_matrix_reps_index_map_;

        // We need 1~(n-1)-th powers of the primitive 2n-th root, m = 2n
        root_powers_ = allocate<complex<double>>(coeff_count, pool_);
        auto host_inv_root_power = HostArray<cuDoubleComplex>(coeff_count);
        // Powers of the primitive 2n-th root have 4-fold symmetry
        if (m >= 8)
        {
            complex_roots_ = make_shared<util::ComplexRoots>(util::ComplexRoots(static_cast<size_t>(m), pool_));
            for (size_t i = 1; i < coeff_count; i++)
            {
                root_powers_[i] = complex_roots_->get_root(reverse_bits(i, logn));
                auto com = complex_roots_->get_root(reverse_bits(i - 1, logn) + 1);
                host_inv_root_power[i] = make_cuDoubleComplex (com.real(), -com.imag());
            }
        }
        else if (m == 4)
        {
            root_powers_[1] = { 0, 1 };
            host_inv_root_power[1] = { 0, -1 };
        }
        inv_root_powers_ = host_inv_root_power;

        temp_values_.resize(slots_);
        temp_com_values_.resize(coeff_count);

        complex_arith_ = ComplexArith();
        fft_handler_ = FFTHandler(complex_arith_);
    }

    __global__ void g_set_conj_values_double(
            const double* values,
            size_t values_size,
            size_t slots,
            cuDoubleComplex* conj_values,
            const uint64_t* matrix_reps_index_map
    ) {
        size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        auto value = tid >= values_size ? 0 : values[tid];
        conj_values[matrix_reps_index_map[tid]] = {value, 0};
        conj_values[matrix_reps_index_map[tid + slots]] = {value, -0};
    }

    __global__ void g_coeff_modulus_reduce_64(
            cuDoubleComplex* conj_values,
            size_t n,
            size_t coeff_modulus_size,
            const Modulus* coeff_modulus,
            uint64_t* destination
    ) {
        size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= (n)) {
            return;
        }
        double coeff_d = round(conj_values[tid].x);
        bool is_negative = coeff_d < 0;
        auto coeff_u = static_cast<uint64_t>(abs(coeff_d));
        for (int i = 0; i < coeff_modulus_size; ++i) {
            if (is_negative) {
                destination[tid + i * n] = kernel_util::d_negate_uint_mod(
                        kernel_util::d_barrett_reduce_64(coeff_u, coeff_modulus[i]), coeff_modulus[i]
                );
            } else {
                destination[tid + i * n] = kernel_util::d_barrett_reduce_64(coeff_u, coeff_modulus[i]);
            }
        }
    }

    template<unsigned l, unsigned n>
    __global__ void fft_inner(cuDoubleComplex *values, cuDoubleComplex *roots, double scalar) {

        auto global_tid = blockIdx.x * 1024 + threadIdx.x;
        auto step = (n / l) / 2;
        auto psi_step = global_tid / step;
        auto target_index = psi_step * step * 2 + global_tid % step;

        auto u = values[target_index];
        auto v = values[target_index + step];
        if (global_tid == 0) {
            printf("step = %u\n", step);
            printf("u = {x = %lf, y = %lf\n", u.x, u.y);
            printf("v = {x = %lf, y = %lf\n", v.x, v.y);
        }
        values[target_index] = cuCadd(u, v);
        values[target_index + step] = cuCmul(cuCsub(u, v), roots[l + psi_step]);
        if (global_tid == 0) {
            printf("x = {x = %lf, y = %lf\n", values[target_index].x, values[target_index].y);
            printf("y = {x = %lf, y = %lf\n", values[target_index + step].x, values[target_index + step].y);
        }
    }

    template<uint l, uint n>
    __global__ void fft_inner_single(cuDoubleComplex *values, cuDoubleComplex *roots, double scalar) {
        auto local_tid = threadIdx.x;

        extern __shared__ cuDoubleComplex shared_array[];

#pragma unroll
        for (uint iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++) {
            auto global_tid = local_tid + iteration_num * 1024;
            shared_array[global_tid] = values[global_tid + blockIdx.x * (n / l)];
        }

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

                auto u = shared_array[target_index];
                auto v = shared_array[target_index + step];
                shared_array[target_index] = cuCadd(u, v);
                shared_array[target_index + step] = cuCmul(cuCsub(u, v), roots[length + psi_step]);
            }
            __syncthreads();
        }

#pragma unroll
        for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++) {
            auto global_tid = local_tid + iteration_num * 1024;

            shared_array[global_tid].x *= scalar;
            shared_array[global_tid].y *= scalar;

            values[global_tid + blockIdx.x * (n / l)] = shared_array[global_tid];
        }
    }

    void fft_transform_to_rev(cuDoubleComplex *operand, size_t coeff_count, cuDoubleComplex *roots, double scalar) {
        switch (coeff_count) {
            case 32768: {
                fft_inner<1, 32768><<<32768 / 1024 / 2, 1024>>>(operand, roots, scalar);
                fft_inner<2, 32768><<<32768 / 1024 / 2, 1024>>>(operand, roots, scalar);
                fft_inner<4, 32768><<<32768 / 1024 / 2, 1024>>>(operand, roots, scalar);
                fft_inner<8, 32768><<<32768 / 1024 / 2, 1024>>>(operand, roots, scalar);
                fft_inner_single<16, 32768><<<16, 1024, 2048 * sizeof(cuDoubleComplex)>>>(operand, roots, scalar);
                break;
            }
            case 16384: {
                fft_inner<1, 16384><<<16384 / 1024 / 2, 1024>>>(operand, roots, scalar);
                fft_inner<2, 16384><<<16384 / 1024 / 2, 1024>>>(operand, roots, scalar);
                fft_inner<4, 16384><<<16384 / 1024 / 2, 1024>>>(operand, roots, scalar);
                fft_inner_single<8, 16384><<<8, 1024, 2048 * sizeof(cuDoubleComplex)>>>(operand, roots, scalar);
                break;
            }
            case 8192: {
                fft_inner<1, 8192><<<8192 / 1024 / 2, 1024>>>(operand, roots, scalar);
                fft_inner<2, 8192><<<8192 / 1024 / 2, 1024>>>(operand, roots, scalar);
                fft_inner_single<4, 8192><<<4, 1024, 2048 * sizeof(cuDoubleComplex)>>>(operand, roots, scalar);
                break;
            }
            case 4096: {
                fft_inner<1, 4096><<<4096 / 1024 / 2, 1024>>>(operand, roots, scalar);
                fft_inner_single<2, 4096> <<<2, 1024, 2048 * sizeof(cuDoubleComplex)>>>(operand, roots, scalar);
                break;
            }
            case 2048: {
                fft_inner_single<1, 2048> <<<1, 1024, 2048 * sizeof(cuDoubleComplex)>>>(operand, roots, scalar);
                break;
            }
            default:
                throw std::invalid_argument("not support");
        }
        CHECK(cudaGetLastError());
    }

    template<unsigned l, unsigned n>
    __global__ void ifft_inner(cuDoubleComplex *values, cuDoubleComplex *roots, double scalar) {

        auto global_tid = blockIdx.x * 1024 + threadIdx.x;
        auto step = (n / l) / 2;
        auto psi_step = global_tid / step;
        auto target_index = psi_step * step * 2 + global_tid % step;

        auto offset = n;
#pragma unroll
        for (uint i = l; i >= 1; i /= 2) {
            offset -= i;
        }

        auto u = values[target_index];
        auto v = values[target_index + step];
        values[target_index] = cuCadd(u, v);
        values[target_index + step] = cuCmul(cuCsub(u, v), roots[offset + psi_step]);
    }

    template<uint l, uint n>
    __global__ void ifft_inner_single(cuDoubleComplex *values, cuDoubleComplex *roots, double scalar) {
        auto local_tid = threadIdx.x;

        extern __shared__ cuDoubleComplex shared_array[];

#pragma unroll
        for (uint iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++) {
            auto global_tid = local_tid + iteration_num * 1024;
            shared_array[global_tid] = values[global_tid + blockIdx.x * (n / l)];
        }
        __syncthreads();
        auto step = 1;
        uint offset = 1;
#pragma unroll
        for (uint length = (n / 2); length >= l; length >>= 1) {

#pragma unroll
            for (uint iteration_num = 0; iteration_num < (n / 1024 / l) / 2; iteration_num++) {
                auto global_tid = local_tid + iteration_num * 1024;
                auto psi_step = global_tid / step;
                auto target_index = psi_step * step * 2 + global_tid % step;
                psi_step = (global_tid + blockIdx.x * (n / l / 2)) / step;

                auto u = shared_array[target_index];
                auto v = shared_array[target_index + step];
                shared_array[target_index] = cuCadd(u, v);
                shared_array[target_index + step] = cuCmul(cuCsub(u, v), roots[offset + psi_step]);
            }
            step <<= 1;
            offset += length;
            __syncthreads();
        }

#pragma unroll
        for (int iteration_num = 0; iteration_num < (n / 1024 / l); iteration_num++) {
            auto global_tid = local_tid + iteration_num * 1024;

            shared_array[global_tid].x *= scalar;
            shared_array[global_tid].y *= scalar;

            values[global_tid + blockIdx.x * (n / l)] = shared_array[global_tid];
        }
    }

    void fft_transform_from_rev(cuDoubleComplex *operand, size_t coeff_count, cuDoubleComplex *roots, double scalar) {
        switch (coeff_count) {
            case 32768: {
                ifft_inner_single<16, 32768><<<16, 1024, 2048 * sizeof(cuDoubleComplex)>>>(operand, roots, scalar);
                ifft_inner<8, 32768><<<32768 / 1024 / 2, 1024>>>(operand, roots, scalar);
                ifft_inner<4, 32768><<<32768 / 1024 / 2, 1024>>>(operand, roots, scalar);
                ifft_inner<2, 32768><<<32768 / 1024 / 2, 1024>>>(operand, roots, scalar);
                ifft_inner<1, 32768><<<32768 / 1024 / 2, 1024>>>(operand, roots, scalar);
                break;
            }
            case 16384: {
                ifft_inner_single<8, 16384><<<8, 1024, 2048 * sizeof(cuDoubleComplex)>>>(operand, roots, scalar);
                ifft_inner<4, 16384><<<16384 / 1024 / 2, 1024>>>(operand, roots, scalar);
                ifft_inner<2, 16384><<<16384 / 1024 / 2, 1024>>>(operand, roots, scalar);
                ifft_inner<1, 16384><<<16384 / 1024 / 2, 1024>>>(operand, roots, scalar);
                break;
            }
            case 8192: {
                ifft_inner_single<4, 8192><<<4, 1024, 2048 * sizeof(cuDoubleComplex)>>>(operand, roots, scalar);
                ifft_inner<2, 8192><<<8192 / 1024 / 2, 1024>>>(operand, roots, scalar);
                ifft_inner<1, 8192><<<8192 / 1024 / 2, 1024>>>(operand, roots, scalar);
                break;
            }
            case 4096: {
                ifft_inner_single<2, 4096> <<<2, 1024, 2048 * sizeof(cuDoubleComplex)>>>(operand, roots, scalar);
                ifft_inner<1, 4096><<<4096 / 1024 / 2, 1024>>>(operand, roots, scalar);
                break;
            }
            case 2048: {
                ifft_inner_single<1, 2048> <<<1, 1024, 2048 * sizeof(cuDoubleComplex)>>>(operand, roots, scalar);
                break;
            }
            default:
                throw std::invalid_argument("not support");
        }
        CHECK(cudaGetLastError());
    }

    void CKKSEncoder::encode_internal_cu(
            const double *values, size_t values_size, parms_id_type parms_id, double scale, Plaintext &destination) {
//        auto time_start0 = std::chrono::high_resolution_clock::now();
        // Verify parameters.
        auto context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw std::invalid_argument("parms_id is not valid for encryption parameters");
        }
        if (!values && values_size > 0)
        {
            throw std::invalid_argument("values cannot be null");
        }
        if (values_size > slots_)
        {
            throw std::invalid_argument("values_size is too large");
        }

        auto &context_data = *context_data_ptr;
        auto &params = context_data.parms();
        auto &coeff_modulus = params.device_coeff_modulus();
        std::size_t coeff_modulus_size = coeff_modulus.size();
        std::size_t coeff_count = params.poly_modulus_degree();

        // Quick sanity check
        if (!util::product_fits_in(coeff_modulus_size, coeff_count))
        {
            throw std::logic_error("invalid parameters");
        }

        // Check that scale is positive and not too large
        if (scale <= 0 || (static_cast<int>(log2(scale)) + 1 >= context_data.total_coeff_modulus_bit_count()))
        {
            throw std::invalid_argument("scale out of bounds");
        }

        auto ntt_tables = context_data.device_small_ntt_tables();

        // values_size is guaranteed to be no bigger than slots_
        std::size_t n = util::mul_safe(slots_, std::size_t(2));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        KernelProvider::copy(temp_values_.get(), values, values_size);

        cudaEventRecord(stop);

//        auto time_end1 = std::chrono::high_resolution_clock::now();
//        auto time_diff1 = std::chrono::duration_cast<std::chrono::microseconds >(time_end1 - time_start0);
//        std::cout << "encode inner file device_com_values end [" << time_diff1.count() << " microseconds]" << std::endl;

        g_set_conj_values_double<<<slots_ / 1024, 1024>>>(
                temp_values_.get(),
                values_size,
                slots_,
                temp_com_values_.get(),
                matrix_reps_index_map_.get()
        );

        double fix = scale / static_cast<double>(n);
        fft_transform_from_rev(temp_com_values_.get(), n, inv_root_powers_.get(), fix);

        // Resize destination to appropriate size
        // Need to first set parms_id to zero, otherwise resize
        // will throw an exception.
        destination.parms_id() = parms_id_zero;

        auto dest_size = util::mul_safe(coeff_count, coeff_modulus_size);

        destination.device_resize(dest_size);

        g_coeff_modulus_reduce_64<<<n / 1024, 1024>>>(
                temp_com_values_.get(),
                n,
                coeff_modulus_size,
                coeff_modulus.get(),
                destination.device_data()
        );

        // Transform to NTT domain
        for (std::size_t i = 0; i < coeff_modulus_size; i++) {
            kernel_util::g_ntt_negacyclic_harvey(destination.device_data() + i * coeff_count, coeff_count, ntt_tables.get()[i]);
        }

        destination.parms_id() = parms_id;
        destination.scale() = scale;

//        auto time_end0 = std::chrono::high_resolution_clock::now();
//        auto time_diff0 = std::chrono::duration_cast<std::chrono::microseconds >(time_end0 - time_start0);
//        std::cout << "encode inner file end [" << time_diff0.count() << " microseconds]" << std::endl;
    }

    void CKKSEncoder::encode_internal(
        double value, parms_id_type parms_id, double scale, Plaintext &destination, MemoryPoolHandle pool) const
    {
        // Verify parameters.
        auto context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();

        // Quick sanity check
        if (!product_fits_in(coeff_modulus_size, coeff_count))
        {
            throw logic_error("invalid parameters");
        }

        // Check that scale is positive and not too large
        if (scale <= 0 || (static_cast<int>(log2(scale)) >= context_data.total_coeff_modulus_bit_count()))
        {
            throw invalid_argument("scale out of bounds");
        }

        // Compute the scaled value
        value *= scale;

        int coeff_bit_count = static_cast<int>(log2(fabs(value))) + 2;
        if (coeff_bit_count >= context_data.total_coeff_modulus_bit_count())
        {
            throw invalid_argument("encoded value is too large");
        }

        double two_pow_64 = pow(2.0, 64);

        // Resize destination to appropriate size
        // Need to first set parms_id to zero, otherwise resize
        // will throw an exception.
        destination.parms_id() = parms_id_zero;
        destination.resize(coeff_count * coeff_modulus_size);

        double coeffd = round(value);
        bool is_negative = signbit(coeffd);
        coeffd = fabs(coeffd);

        // Use faster decomposition methods when possible
        if (coeff_bit_count <= 64)
        {
            uint64_t coeffu = static_cast<uint64_t>(fabs(coeffd));

            if (is_negative)
            {
                for (size_t j = 0; j < coeff_modulus_size; j++)
                {
                    fill_n(
                        destination.data() + (j * coeff_count), coeff_count,
                        negate_uint_mod(barrett_reduce_64(coeffu, coeff_modulus[j]), coeff_modulus[j]));
                }
            }
            else
            {
                for (size_t j = 0; j < coeff_modulus_size; j++)
                {
                    fill_n(
                        destination.data() + (j * coeff_count), coeff_count,
                        barrett_reduce_64(coeffu, coeff_modulus[j]));
                }
            }
        }
        else if (coeff_bit_count <= 128)
        {
            uint64_t coeffu[2]{ static_cast<uint64_t>(fmod(coeffd, two_pow_64)),
                                static_cast<uint64_t>(coeffd / two_pow_64) };

            if (is_negative)
            {
                for (size_t j = 0; j < coeff_modulus_size; j++)
                {
                    fill_n(
                        destination.data() + (j * coeff_count), coeff_count,
                        negate_uint_mod(barrett_reduce_128(coeffu, coeff_modulus[j]), coeff_modulus[j]));
                }
            }
            else
            {
                for (size_t j = 0; j < coeff_modulus_size; j++)
                {
                    fill_n(
                        destination.data() + (j * coeff_count), coeff_count,
                        barrett_reduce_128(coeffu, coeff_modulus[j]));
                }
            }
        }
        else
        {
            // Slow case
            auto coeffu(allocate_uint(coeff_modulus_size, pool));

            // We are at this point guaranteed to fit in the allocated space
            set_zero_uint(coeff_modulus_size, coeffu.get());
            auto coeffu_ptr = coeffu.get();
            while (coeffd >= 1)
            {
                *coeffu_ptr++ = static_cast<uint64_t>(fmod(coeffd, two_pow_64));
                coeffd /= two_pow_64;
            }

            // Next decompose this coefficient
            context_data.rns_tool()->base_q()->decompose(coeffu.get(), pool);

            // Finally replace the sign if necessary
            if (is_negative)
            {
                for (size_t j = 0; j < coeff_modulus_size; j++)
                {
                    fill_n(
                        destination.data() + (j * coeff_count), coeff_count,
                        negate_uint_mod(coeffu[j], coeff_modulus[j]));
                }
            }
            else
            {
                for (size_t j = 0; j < coeff_modulus_size; j++)
                {
                    fill_n(destination.data() + (j * coeff_count), coeff_count, coeffu[j]);
                }
            }
        }

        destination.parms_id() = parms_id;
        destination.scale() = scale;
    }

    void CKKSEncoder::encode_internal(int64_t value, parms_id_type parms_id, Plaintext &destination) const
    {
        // Verify parameters.
        auto context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }

        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();

        // Quick sanity check
        if (!product_fits_in(coeff_modulus_size, coeff_count))
        {
            throw logic_error("invalid parameters");
        }

        int coeff_bit_count = get_significant_bit_count(static_cast<uint64_t>(llabs(value))) + 2;
        if (coeff_bit_count >= context_data.total_coeff_modulus_bit_count())
        {
            throw invalid_argument("encoded value is too large");
        }

        // Resize destination to appropriate size
        // Need to first set parms_id to zero, otherwise resize
        // will throw an exception.
        destination.parms_id() = parms_id_zero;
        destination.resize(coeff_count * coeff_modulus_size);

        if (value < 0)
        {
            for (size_t j = 0; j < coeff_modulus_size; j++)
            {
                uint64_t tmp = static_cast<uint64_t>(value);
                tmp += coeff_modulus[j].value();
                tmp = barrett_reduce_64(tmp, coeff_modulus[j]);
                fill_n(destination.data() + (j * coeff_count), coeff_count, tmp);
            }
        }
        else
        {
            for (size_t j = 0; j < coeff_modulus_size; j++)
            {
                uint64_t tmp = static_cast<uint64_t>(value);
                tmp = barrett_reduce_64(tmp, coeff_modulus[j]);
                fill_n(destination.data() + (j * coeff_count), coeff_count, tmp);
            }
        }

        destination.parms_id() = parms_id;
        destination.scale() = 1.0;
    }

    void CKKSEncoder::encode_internal(const double *values, size_t values_size, parms_id_type parms_id, double scale,
                                      Plaintext &destination, MemoryPoolHandle pool) {
        encode_internal_cu(values, values_size, parms_id, scale, destination);
    }

    void CKKSEncoder::encode_internal(const std::complex<double> *values, size_t values_size, parms_id_type parms_id,
                                      double scale, Plaintext &destination, MemoryPoolHandle pool) const {
//        encode_internal_cu(values, values_size, parms_id, scale, destination, pool);
    }
} // namespace sigma
