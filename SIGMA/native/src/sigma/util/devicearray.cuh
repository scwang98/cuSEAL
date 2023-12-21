#pragma once


#include "hostarray.h"
#include "pointer.h"
#include "../kernelprovider.h"
#include "cuda_runtime.h"
#include <vector>
#include <exception>


namespace sigma::util {

    template<typename T>
    class DevicePointer;

    template<typename T>
    class DeviceArray;

    template<typename T>
    class ConstDevicePointer {
        const T *ptr;
    public:
        ConstDevicePointer() : ptr(nullptr) {}

        explicit ConstDevicePointer(const T *ptr) : ptr(ptr) {}

        explicit ConstDevicePointer(const DevicePointer<T> &d) : ptr(d.get()) {}

        bool isNull() const { return ptr == nullptr; }

        operator bool() const {
            return !isNull();
        }

        const T *get() const { return ptr; }

        ConstDevicePointer<T> operator+(size_t d) {
            return ConstDevicePointer(ptr + d);
        }

        explicit ConstDevicePointer(const DeviceArray<T> &arr) :
                ptr(arr.get()) {}
    };

    template<typename T>
    class DevicePointer {
        friend class DeviceArray<T>;

        T *ptr;
    public:
        DevicePointer(T *ptr) : ptr(ptr) {}

        DevicePointer(DeviceArray<T> &r) : ptr(r.get()) {}

        DevicePointer() {
            ptr = nullptr;
        }

        bool isNull() { return ptr == nullptr; }

        operator bool() const {
            return !isNull();
        }

        const T *get() const { return ptr; }

        T *get() { return ptr; }

        DevicePointer<T> operator+(size_t d) { return DevicePointer(ptr + d); }

        DevicePointer<T> &operator+=(size_t d) {
            ptr += d;
            return *this;
        }
    };

    template<typename T>
    class DeviceArray {
        T *data_;
        size_t len_;
    public:
        DeviceArray() {
            data_ = nullptr;
            len_ = 0;
        }

        explicit DeviceArray(size_t cnt) {
            data_ = KernelProvider::malloc<T>(cnt);
            len_ = cnt;
        }

        SIGMA_NODISCARD size_t length() const {
            return len_;
        }

        SIGMA_NODISCARD size_t size() const {
            return len_;
        }

        // 保留bool allocate编译不通过以提醒
        DeviceArray(T *data, size_t length, bool allocate) {
            if (allocate) {
                len_ = length;
                data_ = KernelProvider::malloc<T>(len_);
                KernelProvider::copy(data_, data, len_);
            } else {
                data_ = data;
                len_ = length;
            }
        }

        DeviceArray(DeviceArray &&a) {
            data_ = a.data_;
            len_ = a.len_;
            a.data_ = nullptr;
            a.len_ = 0;
        }

        DeviceArray &operator=(DeviceArray &&a) {
            if (data_) {
                KernelProvider::free(data_);
            }
            data_ = a.data_;
            len_ = a.len_;
            a.data_ = nullptr;
            a.len_ = 0;
            return *this;
        }

        DeviceArray(const HostArray<T> &host) {
            len_ = host.length();
            data_ = KernelProvider::malloc<T>(len_);
            KernelProvider::copy(data_, host.get(), len_);
        }

        ~DeviceArray() {
            release();
        }

        DeviceArray copy() const {
            T *copied = KernelProvider::malloc<T>(len_);
            KernelProvider::copyOnDevice<T>(copied, data_, len_);
            return DeviceArray(copied, len_);
        }

        DeviceArray &operator=(const DeviceArray &r) {
            if (data_) {
                KernelProvider::free(data_);
            }
            len_ = r.len_;
            data_ = KernelProvider::malloc<T>(len_);
            KernelProvider::copyOnDevice<T>(data_, r.data_, len_);
            return *this;
        }

        DeviceArray(const DeviceArray &r) {
            len_ = r.len_;
            if (len_ > 0) {
                data_ = KernelProvider::malloc<T>(len_);
                KernelProvider::copyOnDevice<T>(data_, r.data_, len_);
            } else {
                data_ = nullptr;
            }
        }

        HostArray<T> toHost() const {
            T *ret = new T[len_];
            KernelProvider::retrieve(ret, data_, len_);
            return HostArray<T>(ret, len_);
        }

        __host__ __device__
        T *get() {
            return data_;
        }

        __host__ __device__
        const T *get() const {
            return data_;
        }

        inline void release() {
            if (data_) {
                KernelProvider::free(data_);
            }
            data_ = nullptr;
            len_ = 0;
        }

        void resize(size_t size) {
            if (len_ == size) {
                return;
            }
            if (data_) {
                KernelProvider::free(data_);
            }
            data_ = KernelProvider::malloc<T>(size);
            len_ = size;
        }

        void set_data(T *data, size_t length) {
            data_ = data;
            len_ = length;
        }

        void set_host_data(const T *data, size_t length) {
            len_ = length;
            data_ = KernelProvider::malloc<T>(len_);
            KernelProvider::copy(data_, data, len_);
        }

        DevicePointer<T> asPointer() { return DevicePointer<T>(data_); }

        ConstDevicePointer<T> asPointer() const { return ConstDevicePointer<T>(data_); }

        DevicePointer<T> operator+(size_t d) {
            return DevicePointer(data_ + d);
        }

        ConstDevicePointer<T> operator+(size_t d) const {
            return ConstDevicePointer(data_ + d);
        }

        __device__ inline T deviceAt(size_t id) const {
            return data_[id];
        }

        __device__ inline T *deviceGet() const {
            return data_;
        }

        T back() const {
            T ret;
            if (data_) KernelProvider::retrieve(&ret, data_ + len_ - 1, 1);
            return ret;
        }

        bool isNull() const {
            return data_ == nullptr;
        }

    };

    template<typename T>
    class DeviceDynamicArray {

        DeviceArray<T> internal;
        size_t size_;

        void move(size_t newCapacity) {
            if (newCapacity == internal.size()) return;
            DeviceArray<T> n(newCapacity);
            if (newCapacity < size_) size_ = newCapacity;
            KernelProvider::copyOnDevice(n.get(), internal.get(), size_);
            if (newCapacity > size_)
                KernelProvider::memsetZero(n.get() + size_, newCapacity - size_);
            internal = std::move(n);
        }

    public:

        DeviceDynamicArray() : internal(), size_(0) {}

        DeviceDynamicArray(size_t len) : internal(len), size_(len) {}

        DeviceDynamicArray(size_t capacity, size_t size) : internal(capacity), size_(size) {}

        DeviceDynamicArray(DeviceArray<T> &&move, size_t size) :
                internal(std::move(move)), size_(size) {}

        DeviceDynamicArray<T> copy() const {
            return DeviceDynamicArray(internal.copy(), size_);
        }

        DeviceDynamicArray(DeviceArray<T> &&move) {
            size_ = move.size();
            internal = std::move(move);
        }

        DeviceDynamicArray(DeviceDynamicArray<T> &&move) {
            size_ = move.size();
            internal = std::move(move.internal);
            move.size_ = 0;
        }

        DeviceDynamicArray(const DeviceDynamicArray<T> &copy) {
            size_ = copy.size();
            internal = std::move(copy.internal.copy());
        }

        DeviceDynamicArray(const HostDynamicArray<T> &h) :
                size_(h.size()),
                internal(h.internal) {}

        DeviceDynamicArray &operator=(const DeviceDynamicArray &copy) {
            if (copy.size_ <= size_) {
                KernelProvider::copyOnDevice(internal.get(), copy.internal.get(), copy.size_);
                size_ = copy.size();
            } else {
                size_ = copy.size();
                internal = std::move(copy.internal.copy());
            }
            return *this;
        }

        DeviceDynamicArray &operator=(DeviceArray<T> &&move) {
            size_ = move.size();
            internal = std::move(move);
            return *this;
        }

        DeviceDynamicArray &operator=(DeviceDynamicArray<T> &&move) {
            size_ = move.size();
            internal = std::move(move.internal);
            move.size_ = 0;
            return *this;
        }

        size_t size() const { return size_; }

        size_t capacity() const { return internal.size(); }

        void reserve(size_t newCapacity) {
            if (capacity() >= newCapacity) return;
            move(newCapacity);
        }

        void shrinkToFit() {
            if (capacity() == size_) return;
            move(size_);
        }

        void release() {
            internal = std::move(DeviceArray<T>());
            size_ = 0;
        }

        void resize(size_t newSize) {
            if (newSize > capacity()) move(newSize);
            size_ = newSize;
        }

        DevicePointer<T> ensure(size_t size) {
            if (size > size_) resize(size);
            return asPointer();
        }

        T *get() { return internal.get(); }

        DevicePointer<T> asPointer() { return internal.asPointer(); }

        ConstDevicePointer<T> asPointer() const {
            return internal.asPointer();
        }

        const T *get() const { return internal.get(); }

        inline std::size_t maxSize() const noexcept {
            return (std::numeric_limits<std::size_t>::max)();
        }

        HostDynamicArray<T> toHost() const {
            T *copy = new T[size_];
            KernelProvider::retrieve(copy, internal.get(), size_);
            return HostDynamicArray<T>(std::move(HostArray(copy, size_)));
        }

    };

}

