#include <variant/variant.h>
#include <cuda_runtime.h>
#include <iostream>
#include "core/cuda_utils.cuh"

struct Type1 {
    int a, b;
    mutable int result;

    __host__ __device__ Type1(int _a = 0, int _b = 0): a(_a), b(_b) {}

    __host__ __device__ int operation(int opr1, int opr2) const {
        result = (a + opr1) * (b + opr2);
        return result;
    }
};

struct Type2 {
    int a, b;
    mutable int result;

    __host__ __device__ Type2(int _a = 0, int _b = 0): a(_a), b(_b) {}

    __host__ __device__ int operation(int opr1, int opr2) const {
        result = a * opr1 + b * opr2;
        return result;
    }
};
struct TypeVisitor {
    int opr1, opr2;
    __host__ __device__ TypeVisitor(int op1 = 1, int op2 = 1): opr1(op1), opr2(op2) {}
    __host__ __device__ int operator()(const Type1& t) const { return t.operation(opr1, opr2); }
    __host__ __device__ int operator()(const Type2& t) const { return t.operation(opr1, opr2); }
};

using VarType = variant::variant<Type1, Type2>;

__global__ void kernel_op(VarType* objects, int* result_buffer) {
    result_buffer[threadIdx.x] = variant::apply_visitor(TypeVisitor(), objects[threadIdx.x]);
}

int main() {
    VarType* vars;
    CUDA_CHECK_RETURN(cudaMallocManaged(&vars, sizeof(VarType) * 16));
    vars[0] = Type1(0, 1);
    vars[1] = Type2(0, 1);
    vars[2] = Type1(1, 1);
    vars[3] = Type2(1, 1);

    vars[4] = Type1(2, 1);
    vars[5] = Type2(2, 1);
    vars[6] = Type1(1, 2);
    vars[7] = Type2(1, 2);

    vars[8]  = Type1(2, 2);
    vars[9]  = Type1(2, 3);
    vars[10] = Type1(2, 4);
    vars[11] = Type2(3, 4);

    vars[12] = Type2(2, 2);
    vars[13] = Type2(2, 3);
    vars[14] = Type2(2, 4);
    vars[15] = Type1(3, 4);

    int* res_buffer;
    CUDA_CHECK_RETURN(cudaMallocManaged(&res_buffer, sizeof(int) * 16));
    kernel_op<<<1, 16>>>(vars, res_buffer);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    for (int i = 0; i < 16; i++)
        printf("%d\n", res_buffer[i]);
    for (int i = 0; i < 16; i++)
        printf("var[%d].result = %d\n", i, *((int*)(&vars[i]) + 2));
    CUDA_CHECK_RETURN(cudaFree(vars));
    CUDA_CHECK_RETURN(cudaFree(res_buffer));
    return 0;
}