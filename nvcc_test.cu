#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "sharedMemPerBlock: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "sharedMemPerMultiprocessor: " << prop.sharedMemPerMultiprocessor << " bytes" << std::endl;
    std::cout << "sharedMemPerBlockOptin: " << prop.sharedMemPerBlockOptin << " bytes" << std::endl;
    std::cout << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "concurrentKernels: " << prop.concurrentKernels << std::endl;
    std::cout << "maxBlocksPerMultiProcessor: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "multiProcessorCount: " << prop.multiProcessorCount << std::endl;
    std::cout << "totalConstMem: " << prop.totalConstMem << " bytes"<< std::endl;
    return 0;
}