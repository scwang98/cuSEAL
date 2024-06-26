#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cinttypes>
#include <string>
#include <cmath>

class uint128_ntt
{
public:

	uint64_t low;
	uint64_t high;

	__host__ __device__ __forceinline__ uint128_ntt()
	{
		low = 0;
		high = 0;
	}

	__host__ __device__ __forceinline__ uint128_ntt(const uint64_t& x)
	{
		low = x;
		high = 0;
	}

	__host__ __device__ __forceinline__ void operator=(const uint128_ntt& r)
	{
		low = r.low;
		high = r.high;
	}

	__host__ __device__ __forceinline__ void operator=(const uint64_t& r)
	{
		low = r;
		high = 0;
	}

	__host__ __device__ __forceinline__ uint128_ntt operator<<(const unsigned& shift)
	{
		uint128_ntt z;

		z.high = high << shift;
		z.high = (low >> (64 - shift)) | z.high;
		z.low = low << shift;

		return z;
	}

	__host__ __device__ __forceinline__ uint128_ntt operator>>(const unsigned& shift)
	{
		uint128_ntt z;

		z.low = low >> shift;
		z.low = (high << (64 - shift)) | z.low;
		z.high = high >> shift;

		return z;
	}

	__host__ __device__ __forceinline__ static void shiftr(uint128_ntt& x, const unsigned& shift)
	{
		x.low = x.low >> shift;
		x.low = (x.high << (64 - shift)) | x.low;
		x.high = x.high >> shift;

	}

	__host__ static uint128_ntt exp2(const int& e)
	{
		uint128_ntt z;

		if (e < 64)
			z.low = 1ull << e;
		else
			z.high = 1ull << (e - 64);

		return z;
	}

	__host__ static int log_2(const uint128_ntt& x)
	{
		int z = 0;

		if (x.high != 0)
			z = log2((double)x.high) + 64;
		else
			z = log2((double)x.low);

		return z;
	}

	__host__ __device__ __forceinline__ static int clz(uint128_ntt x)
	{
		unsigned cnt = 0;

		if (x.high == 0)
		{
			while (x.low != 0)
			{
				cnt++;
				x.low = x.low >> 1;
			}

			return 128 - cnt;
		}
		else
		{
			while (x.high != 0)
			{
				cnt++;
				x.high = x.high >> 1;
			}

			return 64 - cnt;
		}
	}

};

__host__ __device__ __forceinline__ static void operator<<=(uint128_ntt& x, const unsigned& shift)
{
	x.low = x.low >> shift;
	x.low = (x.high << (64 - shift)) | x.low;
	x.high = x.high >> shift;

}

__host__ __device__ __forceinline__ bool operator==(const uint128_ntt& l, const uint128_ntt& r)
{
	if ((l.low == r.low) && (l.high == r.high))
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator<(const uint128_ntt& l, const uint128_ntt& r)
{
	if (l.high < r.high)
		return true;
	else if (l.high > r.high)
		return false;
	else if (l.low < r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator<(const uint128_ntt& l, const uint64_t& r)
{
	if (l.high != 0)
		return false;
	else if (l.low > r)
		return false;
	else
		return true;
}

__host__ __device__ __forceinline__ bool operator>(const uint128_ntt& l, const uint128_ntt& r)
{
	if (l.high > r.high)
		return true;
	else if (l.high < r.high)
		return false;
	else if (l.low > r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator<=(const uint128_ntt& l, const uint128_ntt& r)
{
	if (l.high < r.high)
		return true;
	else if (l.high > r.high)
		return false;
	else if (l.low <= r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator>=(const uint128_ntt& l, const uint128_ntt& r)
{
	if (l.high > r.high)
		return true;
	else if (l.high < r.high)
		return false;
	else if (l.low >= r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ uint128_ntt operator+(const uint128_ntt& x, const uint128_ntt& y)
{
	uint128_ntt z;

	z.low = x.low + y.low;
	z.high = x.high + y.high + (z.low < x.low);

	return z;
}

__host__ __device__ __forceinline__ uint128_ntt operator+(const uint128_ntt& x, const uint64_t& y)
{
	uint128_ntt z;

	z.low = x.low + y;
	z.high = x.high + (z.low < x.low);

	return z;
}

__host__ __device__ __forceinline__ uint128_ntt operator-(const uint128_ntt& x, const uint128_ntt& y)
{
	uint128_ntt z;

	z.low = x.low - y.low;
	z.high = x.high - y.high - (x.low < y.low);

	return z;

}

__host__ __device__ __forceinline__ void operator-=(uint128_ntt& x, const uint128_ntt& y)
{
	x.high = x.high - y.high - (x.low < y.low);
	x.low = x.low - y.low;
}

__host__ __device__ __forceinline__ uint128_ntt operator-(const uint128_ntt& x, const uint64_t& y)
{
	uint128_ntt z;

	z.low = x.low - y;
	z.high = x.high - (x.low < y);

	return z;

}

__host__ __device__ __forceinline__ uint128_ntt operator/(uint128_ntt x, const uint64_t& y)
{
	uint128_ntt z;
	uint128_ntt ycomp(y);
	uint128_ntt d(y);

	unsigned shift = uint128_ntt::clz(d) - uint128_ntt::clz(x);

	d = d << shift;

	while (shift != 0)
	{
		shift--;
		z = z << 1;
		if (d <= x)
		{
			x = x - d;
			z = z + 1;
		}
		d = d >> 1;
	}

	z = z << 1;
	if (d <= x)
	{
		x = x - d;
		z = z + 1;
	}
	d = d >> 1;

	return z;
}

__host__ __device__ __forceinline__ uint128_ntt operator%(uint128_ntt x, const uint64_t& y)
{
	if (x < y)
		return x;

	uint128_ntt z;
	uint128_ntt ycomp(y);
	uint128_ntt d(y);

	unsigned shift = uint128_ntt::clz(d) - uint128_ntt::clz(x);

	d = d << shift;

	while (shift != 0)
	{
		shift--;
		z = z << 1;
		if (d <= x)
		{
			x = x - d;
			z = z + 1;
		}
		d = d >> 1;
	}

	z = z << 1;
	if (d <= x)
	{
		x = x - d;
		z = z + 1;
	}
	d = d >> 1;

	return x;
}

__host__ inline static uint128_ntt host64x2(const uint64_t& x, const uint64_t& y)
{
	uint128_ntt z;

	uint128_ntt ux(x);
	uint128_ntt uy(y);

	int shift = 0;

	// hello elementary school
	while (uy.low != 0)
	{
		if (uy.low & 1)
		{
			if (shift == 0)
				z = z + ux;
			else
				z = z + (ux << shift);
		}

		shift++;

		uint128_ntt::shiftr(uy, 1);

	}

	return z;
}

__device__ __forceinline__ void sub128(uint128_ntt& a, const uint128_ntt& b)
{
	asm("{\n\t"
		"sub.cc.u64      %1, %3, %5;    \n\t"
		"subc.u64        %0, %2, %4;    \n\t"
		"}"
		: "=l"(a.high), "=l"(a.low)
		: "l"(a.high), "l"(a.low), "l"(b.high), "l"(b.low));
}

__device__ __forceinline__ void mul64(const uint64_t& a, const uint64_t& b, uint128_ntt& c)
{
	uint4 res;

	asm("{\n\t"
		"mul.lo.u32      %3, %5, %7;    \n\t"
		"mul.hi.u32      %2, %5, %7;    \n\t" //alow * blow
		"mad.lo.cc.u32   %2, %4, %7, %2;\n\t"
		"madc.hi.u32     %1, %4, %7,  0;\n\t" //ahigh * blow
		"mad.lo.cc.u32   %2, %5, %6, %2;\n\t"
		"madc.hi.cc.u32  %1, %5, %6, %1;\n\t" //alow * bhigh
		"madc.hi.u32     %0, %4, %6,  0;\n\t"
		"mad.lo.cc.u32   %1, %4, %6, %1;\n\t" //ahigh * bhigh
		"addc.u32        %0, %0, 0;     \n\t" //add final carry
		"}"
		: "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
		: "r"((unsigned)(a >> 32)), "r"((unsigned)a), "r"((unsigned)(b >> 32)), "r"((unsigned)b));

	c.high = ((uint64_t)res.x << 32) + res.y;
	c.low = ((uint64_t)res.z << 32) + res.w;;
}
