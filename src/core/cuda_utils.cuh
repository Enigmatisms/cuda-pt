#pragma once
#include <omp.h>
#include <chrono>
#include <random>
#include <iostream>
#include <algorithm>
#include <type_traits>

#define CPT_CPU_GPU __host__ __device__
#define CPT_CPU_GPU_INLINE __forceinline__ __host__ __device__
#define CPT_GPU __device__
#define CPT_GPU_INLINE __forceinline__ __device__
#define CPT_CPU __host__
#define CPT_CPU_INLINE __forceinline__ __host__

#define CONDITION_TEMPLATE(VecType, TargetType) \
    template<typename VecType, typename = std::enable_if_t<std::is_same_v<std::decay_t<VecType>, TargetType>>>

#define CONDITION_TEMPLATE_2(T1, T2, TargetType) \
    template<typename T1, typename T2, typename = \
        std::enable_if_t<\
           std::is_same_v<std::decay_t<T1>, TargetType> && \
           std::is_same_v<std::decay_t<T2>, TargetType> \
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

inline int to_int(float x) { return int(powf(std::clamp(x, 0.f, 1.f), 1 / 2.2) * 255 + .5); }
