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

/**
 * @author: Qianyue He
 * @brief Scheduler impl
 * @date: 2025.10.2
 */
#pragma once
#include "core/cuda_utils.cuh"

extern __device__ int tile_count_ptr[1];

class SingleTileScheduler {
  public:
    static constexpr bool is_dynamic = false;
    CPT_GPU SingleTileScheduler(int *const __restrict__ smem_ptr_ = nullptr,
                                int *const __restrict__ gmem_ptr_ = nullptr) {}

    CPT_GPU_INLINE int get_initial_work(int width) const {
        int base = threadIdx.x + blockIdx.x * blockDim.x; // low 16 bit for x
        base |= (threadIdx.y + blockIdx.y * blockDim.y)
                << 16; // high 16 bit for y
        return base;
    }

    CPT_GPU_INLINE int get_next_work(int width) const { return -1; }

    // pos pass in py
    CPT_GPU_INLINE bool is_valid(int &pos, int bound = 0) const {
        return pos >= 0;
    }
};

class PreemptivePersistentTileScheduler {
  private:
    int *const __restrict__ smem_ptr;
    int *const __restrict__ gmem_ptr;

    static CPT_CPU_GPU_INLINE void get_coords(int tile_id, int width, int &px,
                                              int &py) {
        int wtile_num = (width + 31) >> 5; // + 31 / 32
        py = tile_id / wtile_num;
        px = tile_id % wtile_num;

        px = threadIdx.x + px * blockDim.x;
        py = threadIdx.y + py * blockDim.y;
    }

  public:
    static constexpr bool is_dynamic = true;

    CPT_GPU PreemptivePersistentTileScheduler(int *const __restrict__ smem_ptr_,
                                              int *const __restrict__ gmem_ptr_)
        : smem_ptr(smem_ptr_), gmem_ptr(gmem_ptr_) {
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 &&
            threadIdx.y == 0)
            *gmem_ptr = gridDim.x * gridDim.y;
    }

    CPT_GPU_INLINE int get_initial_work(int width) const {
        int tile_id = blockIdx.x * gridDim.y + blockIdx.y;
        int py = 0, px = 0;
        get_coords(tile_id, width, px, py);
        return (py << 16) | px;
    }

    CPT_GPU_INLINE int get_next_work(int width) const {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            *smem_ptr = atomicAdd(gmem_ptr, 1);
        }
        __syncthreads();
        int py = 0, px = 0;
        get_coords(*smem_ptr, width, px, py);
        return (py << 16) | px;
    }

    // bound here is height, pass in image.h()
    CPT_GPU_INLINE bool is_valid(int &pos, int bound = 0) const {
        // test whether py is in range
        return (pos >> 16) < ((bound + 3) & 0xfffffffc);
    }
};
