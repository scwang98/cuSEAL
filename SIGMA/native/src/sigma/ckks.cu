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

        complex_arith_ = ComplexArith();
        fft_handler_ = FFTHandler(complex_arith_);
    }

    __global__ void g_set_conj_values_complex(
            const cuDoubleComplex* values,
            size_t values_size,
            size_t slots,
            cuDoubleComplex* conj_values,
            const uint64_t* matrix_reps_index_map
    ) {
        GET_INDEX_COND_RETURN(values_size);
        auto value = values[gindex];
        conj_values[matrix_reps_index_map[gindex]] = value;
        conj_values[matrix_reps_index_map[gindex + slots]] = cuConj(value);
    }

    __global__ void g_set_conj_values_double(
            const double* values,
            size_t values_size,
            size_t slots,
            cuDoubleComplex* conj_values,
            const uint64_t* matrix_reps_index_map
    ) {
        GET_INDEX_COND_RETURN(values_size);
        auto value = values[gindex];
        conj_values[matrix_reps_index_map[gindex]] = {value, 0};
        conj_values[matrix_reps_index_map[gindex + slots]] = {value, -0};
    }

    inline void set_conj_values(
            const complex<double>* values,
            size_t values_size,
            const size_t slots,
            cuDoubleComplex* conj_values,
            const uint64_t* matrix_reps_index_map
    ) {
        auto device_values = util::DeviceArray<complex<double>>(values_size);
        KernelProvider::copy(device_values.get(), values, values_size);
        KERNEL_CALL(g_set_conj_values_complex, slots)(
                reinterpret_cast<const cuDoubleComplex *>(device_values.get()),
                values_size,
                slots,
                conj_values,
                matrix_reps_index_map
        );
    }

    inline void set_conj_values(
            const double* values,
            size_t values_size,
            size_t slots,
            cuDoubleComplex* conj_values,
            const uint64_t* matrix_reps_index_map
    ) {
        auto start = std::chrono::high_resolution_clock::now();
        auto device_values = util::DeviceArray<double>(values_size);
        KernelProvider::copy(device_values.get(), values, values_size);
        KERNEL_CALL(g_set_conj_values_double, slots)(
                device_values.get(),
                values_size,
                slots,
                conj_values,
                matrix_reps_index_map
        );
    }

    __global__ void gFftTransferFromRevLayered(
            size_t layer,
            cuDoubleComplex* operand,
            size_t poly_modulus_degree_power,
            const cuDoubleComplex * roots
    ) {
        GET_INDEX_COND_RETURN(1 << (poly_modulus_degree_power - 1));
        size_t m = 1 << (poly_modulus_degree_power - 1 - layer);
        size_t gap = 1 << layer;
        size_t rid = (1 << poly_modulus_degree_power) - (m << 1) + 1 + (gindex >> layer);
        size_t coeff_index = ((gindex >> layer) << (layer + 1)) + (gindex & (gap - 1));

        cuDoubleComplex &x = operand[coeff_index];
        cuDoubleComplex &y = operand[coeff_index + gap];

        double ur = x.x, ui = x.y, vr = y.x, vi = y.y;
        double rr = roots[rid].x, ri = roots[rid].y;

        // x = u + v
        x.x = ur + vr;
        x.y = ui + vi;

        // y = (u-v) * r
        ur -= vr; ui -= vi; // u <- u - v
        y.x = ur * rr - ui * ri;
        y.y = ur * ri + ui * rr;
    }

    __global__ void gMultiplyScalar(
            cuDoubleComplex *operand,
            size_t n,
            double scalar
    ) {
        GET_INDEX_COND_RETURN(n);
        operand[gindex].x *= scalar;
        operand[gindex].y *= scalar;
    }

    void kFftTransferFromRev(
            cuDoubleComplex* operand,
            size_t poly_modulus_degree_power,
            const cuDoubleComplex* roots,
            double fix = 1
    ) {
        std::size_t n = size_t(1) << poly_modulus_degree_power;
        std::size_t m = n >> 1; std::size_t layer = 0;
        for(; m >= 1; m >>= 1) {
            KERNEL_CALL(gFftTransferFromRevLayered, n >> 1)(
                    layer,
                    operand,
                    poly_modulus_degree_power,
                    roots
            );
            layer++;
        }
        if (fix != 1) {
            KERNEL_CALL(gMultiplyScalar, n)(operand, n, fix);
        }
    }

    __global__ void gMaxReal(
            double* complexes,
            size_t n_sqrt,
            size_t n,
            double* out
    ) {
        GET_INDEX_COND_RETURN(n_sqrt);
        double m = 0;
        FOR_N(i, n_sqrt) {
            size_t id = gindex * n_sqrt + i;
            if (id >= n) break;
            if (fabs(complexes[id * 2]) > m) m = fabs(complexes[id * 2]);
        }
        out[gindex] = m;
    }

    __global__ void findMax(cuDoubleComplex* d_array, size_t size, double* result) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= size) {
            return;
        }

        int stride = blockDim.x * gridDim.x;

        double localMax = -DBL_MAX;

        for (int i = tid; i < size; i += stride) {
            if (d_array[i].x > localMax) {
                localMax = d_array[i].x;
            }
        }

        // 存储局部最大值到共享内存
        __shared__ double s_max[256]; // 256是一个线程块的最大数量
        s_max[threadIdx.x] = localMax;
        __syncthreads();

        // 在每个线程块内查找局部最大值
        int s = blockDim.x / 2;
        while (s > 0) {
            if (threadIdx.x < s) {
                if (fabs(s_max[threadIdx.x]) < fabs(s_max[threadIdx.x + s])) {
                    s_max[threadIdx.x] = fabs(s_max[threadIdx.x + s]);
                }
            }
            __syncthreads();
            s /= 2;
        }

        // 第一个线程块的线程将局部最大值存储在全局结果中
        if (threadIdx.x == 0) {
            result[blockIdx.x] = s_max[0];
        }
    }

    __global__ void gEncodeInternalComplexArrayUtilA(
            cuDoubleComplex* conj_values,
            size_t n,
            size_t coeff_modulus_size,
            const Modulus* coeff_modulus,
            uint64_t* destination
    ) {
        GET_INDEX_COND_RETURN(n);
        double coeffd = round(conj_values[gindex].x);
        bool is_negative = coeffd < 0;
        uint64_t coeffu = static_cast<uint64_t>(abs(coeffd));
        FOR_N(j, coeff_modulus_size) {
            if (is_negative) {
                destination[gindex + j * n] = kernel_util::dNegateUintMod(
                        kernel_util::dBarrettReduce64(coeffu, coeff_modulus[j]), coeff_modulus[j]
                );
            } else {
                destination[gindex + j * n] = kernel_util::dBarrettReduce64(coeffu, coeff_modulus[j]);
            }
        }
    }

    __global__ void gEncodeInternalComplexArrayUtilB(
            cuDoubleComplex* conj_values,
            size_t n,
            size_t coeff_modulus_size,
            const Modulus* coeff_modulus,
            uint64_t* destination
    ) {
        GET_INDEX_COND_RETURN(n);
        double two_pow_64 = pow(2.0, 64);
        double coeffd = round(conj_values[gindex].x);
        bool is_negative = coeffd < 0;
        coeffd = fabs(coeffd);
        uint64_t coeffu[2] = {
                static_cast<uint64_t>(fmod(coeffd, two_pow_64)),
                static_cast<uint64_t>(coeffd / two_pow_64),
        };
        FOR_N(j, coeff_modulus_size) {
            if (is_negative) {
                destination[gindex + j * n] = kernel_util::dNegateUintMod(
                        kernel_util::dBarrettReduce128(coeffu, coeff_modulus[j]), coeff_modulus[j]
                );
            } else {
                destination[gindex + j * n] = kernel_util::dBarrettReduce128(coeffu, coeff_modulus[j]);
            }
        }
    }

//    template <typename T, typename>
    void CKKSEncoder::encode_internal_cu(
            const double *values, size_t values_size, parms_id_type parms_id, double scale, Plaintext &destination,
            MemoryPoolHandle pool) const
    {
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
        if (!pool)
        {
            throw std::invalid_argument("pool is uninitialized");
        }

        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.device_coeff_modulus();
        std::size_t coeff_modulus_size = coeff_modulus.size();
        std::size_t coeff_count = parms.poly_modulus_degree();

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

        auto device_conj_values = kernel_util::kAllocateZero<cuDoubleComplex>(n);
        auto device_values = util::DeviceArray<double>(values_size);

        auto start = std::chrono::high_resolution_clock::now();

//        set_conj_values(values, values_size, slots_, device_conj_values.get(), matrix_reps_index_map_.get());
        auto device_com_values = util::DeviceArray<double>(values_size);
        KernelProvider::copy(device_com_values.get(), values, values_size);
        {
            KERNEL_CALL(g_set_conj_values_double, slots_)(
                    device_com_values.get(),
                    values_size,
                    slots_,
                    device_conj_values.get(),
                    matrix_reps_index_map_.get()
            );
        }

        double fix = scale / static_cast<double>(n);
        kFftTransferFromRev(
                device_conj_values.get(),
                util::get_power_of_two(n),
                inv_root_powers_.get(),
                fix);

        double max_coeff = 0;
        {
            size_t block_count = kernel_util::ceilDiv_(n, 256);
            int sharedMemSize = 256 * sizeof(double);
            auto* d_result = KernelProvider::malloc<double>(block_count); // 用于存储每个线程块的局部最大值
            double h_result[block_count];
            findMax<<<block_count, 256, sharedMemSize>>>(device_conj_values.get(), n, d_result);
            KernelProvider::retrieve(h_result, d_result, block_count);
            for (int i = 0; i < block_count; i++) {
                if (h_result[i] > max_coeff) {
                    max_coeff = h_result[i];
                }
            }
            KernelProvider::free(d_result);
        }

        // Verify that the values are not too large to fit in coeff_modulus
        // Note that we have an extra + 1 for the sign bit
        // Don't compute logarithmis of numbers less than 1
        int max_coeff_bit_count = static_cast<int>(std::ceil(std::log2(std::max<>(max_coeff, 1.0)))) + 1;
        if (max_coeff_bit_count >= context_data.total_coeff_modulus_bit_count())
        {
            throw std::invalid_argument("encoded values are too large");
        }

        double two_pow_64 = std::pow(2.0, 64);

        // Resize destination to appropriate size
        // Need to first set parms_id to zero, otherwise resize
        // will throw an exception.
        destination.parms_id() = parms_id_zero;

        auto dest_size = util::mul_safe(coeff_count, coeff_modulus_size);

        DeviceArray<uint64_t> dest_arr(dest_size);

        // Use faster decomposition methods when possible
        if (max_coeff_bit_count <= 64)
        {
            KERNEL_CALL(gEncodeInternalComplexArrayUtilA, n) (
                    device_conj_values.get(),
                    n, coeff_modulus_size,
                    coeff_modulus.get(),
                    // TODO: wangshuchao
                    dest_arr.get()
            );
        }
        else if (max_coeff_bit_count <= 128)
        {
            KERNEL_CALL(gEncodeInternalComplexArrayUtilB, n) (
                    device_conj_values.get(),
                    n,
                    coeff_modulus_size,
                    coeff_modulus.get(),
                    dest_arr.get()
            );
        }
        else
        {
            // Slow case
            throw std::invalid_argument("not support");
        }

        // Transform to NTT domain
        for (std::size_t i = 0; i < coeff_modulus_size; i++) {
            kernel_util::g_ntt_negacyclic_harvey(dest_arr.get() + i * coeff_count, coeff_count, ntt_tables.get()[i]);
        }

        destination.resize(dest_size);
        cudaMemcpy(destination.data(), dest_arr.get(), dest_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        destination.parms_id() = parms_id;
        destination.scale() = scale;
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
                                      Plaintext &destination, MemoryPoolHandle pool) const {
        encode_internal_cu(values, values_size, parms_id, scale, destination, pool);
    }

    void CKKSEncoder::encode_internal(const std::complex<double> *values, size_t values_size, parms_id_type parms_id,
                                      double scale, Plaintext &destination, MemoryPoolHandle pool) const {
//        encode_internal_cu(values, values_size, parms_id, scale, destination, pool);
    }
} // namespace sigma
