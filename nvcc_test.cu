// Copyright (C) 2025 Qianyue He
//
// This program is free software: you can redistribute it and/or
// modify it under the terms of the GNU Affero General Public License
// as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
// the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General
// Public License along with this program. If not, see
//
//             <https://www.gnu.org/licenses/>.

#include <cuda_runtime.h>
#include <iostream>

class Vec3 {
  private:
    float3 _data;

  public:
    __host__ __device__ Vec3() {}
    constexpr __host__ __device__ Vec3(float _x, float _y, float _z)
        : _data({_x, _y, _z}) {}

    constexpr __host__ __device__ const float &x() const { return _data.x; }
    constexpr __host__ __device__ const float &y() const { return _data.y; }
    constexpr __host__ __device__ const float &z() const { return _data.z; }
};

int main(int argc, char **argv) {
    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "sharedMemPerBlock: " << prop.sharedMemPerBlock << " bytes"
              << std::endl;
    std::cout << "sharedMemPerMultiprocessor: "
              << prop.sharedMemPerMultiprocessor << " bytes" << std::endl;
    std::cout << "sharedMemPerBlockOptin: " << prop.sharedMemPerBlockOptin
              << " bytes" << std::endl;
    std::cout << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "maxThreadsPerMultiProcessor: "
              << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "concurrentKernels: " << prop.concurrentKernels << std::endl;
    std::cout << "maxBlocksPerMultiProcessor: "
              << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "multiProcessorCount: " << prop.multiProcessorCount
              << std::endl;
    std::cout << "totalConstMem: " << prop.totalConstMem << " bytes"
              << std::endl;
    std::cout << "multiProcessorCount: " << prop.multiProcessorCount
              << std::endl;

    constexpr Vec3 data(1, 2, 3);
    std::cout << data.x() << ", " << data.y() << ", " << data.z() << std::endl;

    return 0;
}
