#pragma once
#include <omp.h>
#include <chrono>
#include <random>
#include <iostream>

#define CPT_CPU_GPU __host__ __device__
#define CPT_GPU __device__
#define CPT_CPU __host__

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

class TicToc {
private:
    std::chrono::system_clock::time_point tp;
public:
    void tic() {
        tp = std::chrono::system_clock::now();
    }

    double toc() const {
        auto dur = std::chrono::system_clock::now() - tp;
        auto count = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
        return static_cast<double>(count) / 1e3;
    }
};