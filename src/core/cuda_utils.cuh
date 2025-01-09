#pragma once
#include <omp.h>
#include <chrono>
#include <random>
#include <iostream>
#include <algorithm>
#include <type_traits>

#define CPT_KERNEL __global__
#define CPT_CPU_GPU __host__ __device__
#define CPT_CPU_GPU_INLINE __forceinline__ __host__ __device__
#define CPT_GPU __device__
#define CPT_GPU_INLINE __forceinline__ __device__
#define CPT_CPU __host__
#define CPT_CPU_INLINE __forceinline__ __host__
#define CPT_GPU_CONST __constant__
#define CUDA_PT_SINGLE_PADDING(id) int __bytes##id;

#define FLOAT4(v) (*(reinterpret_cast<float4*>(&v)))
#define FLOAT2(v) (*(reinterpret_cast<float2*>(&v)))
#define CONST_FLOAT4(v) (*(reinterpret_cast<const float4*>(&v)))
#define CONST_FLOAT2(v) (*(reinterpret_cast<const float2*>(&v)))

#define UINT4(v) (*(reinterpret_cast<uint4*>(&v)))
#define UINT2(v) (*(reinterpret_cast<uint2*>(&v)))
#define CONST_UINT4(v) (*(reinterpret_cast<const uint4*>(&v)))
#define CONST_UINT2(v) (*(reinterpret_cast<const uint2*>(&v)))
#define INT_REF_CAST(v) reinterpret_cast<int&>(v)
#define INT_CREF_CAST(v) reinterpret_cast<const int&>(v)

#define CONDITION_TEMPLATE(VecType, TargetType) \
    template<typename VecType, typename = std::enable_if_t<std::is_same_v<std::decay_t<VecType>, TargetType>>>

#define CONDITION_TEMPLATE_DEFAULT(VecType, TargetType) \
    template<typename VecType, std::enable_if_t<std::is_same_v<std::decay_t<VecType>, TargetType>, int> = 0>

#define CONDITION_TEMPLATE_2(T1, T2, TargetType) \
    template<typename T1, typename T2, typename = \
        std::enable_if_t<\
           std::is_same_v<std::decay_t<T1>, TargetType> && \
           std::is_same_v<std::decay_t<T2>, TargetType> \
        >>

#define CONDITION_TEMPLATE_SEP_2(T1, T2, TargetType1, TargetType2) \
    template<typename T1, typename T2, typename = \
        std::enable_if_t<\
           std::is_same_v<std::decay_t<T1>, TargetType1> && \
           std::is_same_v<std::decay_t<T2>, TargetType2> \
        >>

#define CONDITION_TEMPLATE_SEP_3(T1, T2, T3, TargetType1, TargetType2, TargetType3) \
    template<typename T1, typename T2, typename T3, typename = \
        std::enable_if_t<\
           std::is_same_v<std::decay_t<T1>, TargetType1> && \
           std::is_same_v<std::decay_t<T2>, TargetType2> && \
           std::is_same_v<std::decay_t<T3>, TargetType3> \
        >>

#ifndef NO_CUDA
__host__ static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__host__ static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	printf("%s returned %s(%d) at %s:%u\n", statement, cudaGetErrorString(err), err, file, line);
	exit (1);
}
#endif

template <typename T> 
CPT_CPU_GPU_INLINE T sgn(T val) {
    return (T(0) < val) - (val < T(0));
}


template<typename T1, typename T2>
CPT_CPU_GPU_INLINE auto select(T1&& v_true, T2&& v_false, bool predicate) {
    return v_true * predicate + v_false * (1 - predicate);
}

CPT_CPU_INLINE int to_int_linear(float x) { return int(std::clamp(x, 0.f, 1.f) * 255.f); }
CPT_CPU_INLINE int to_int(float x) { return int(powf(std::clamp(x, 0.f, 1.f), 1.f / 2.1f) * 255.f + .5f); }

CPT_CPU_GPU_INLINE float roughness_to_alpha(float roughness) {
    roughness = fmaxf(roughness, 1e-3f);
    float x = logf(roughness);
    return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
        0.000640711f * x * x * x * x;
}

CPT_GPU_INLINE int float_to_ordered_int( float float_v ) {
    int int_v = __float_as_int( float_v );
    return (int_v >= 0 ) ? int_v : int_v ^ 0x7FFFFFFF;
}
CPT_GPU_INLINE float ordered_int_to_float( int int_v ) {
    return __int_as_float( (int_v >= 0) ? int_v : int_v ^ 0x7FFFFFFF);
}