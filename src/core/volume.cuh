/**
 * @file volume.cuh
 * @author your name (you@domain.com)
 * @brief Volume definition
 * @version 0.1
 * @date 2025-02-02
 * @copyright Copyright (c) 2025
 */

#include "core/aabb.cuh"
#include "core/vec4.cuh"

/**
 * This is a SoA containing all the 
 * infos regarding volumes (homogeneous or heterogeneous)
 * 5 volumes at most (4 different non-zero offsets)
 */
class DeviceVolumes {
private:
    // POD, compacted Index Bound
    struct IndexBound {
        uint32_t _data;

        CPT_CPU_GPU_INLINE uint32_t x() const noexcept { return _data >> 21; }                      // x 0-> 2047
        CPT_CPU_GPU_INLINE uint32_t y() const noexcept { return (_data >> 10) & 0x00003fff; }       // y 0-> 2047
        CPT_CPU_GPU_INLINE uint32_t z() const noexcept { return _data & 0x00001fff; }               // z 0-> 1023

        CPT_CPU_GPU_INLINE void setup(uint32_t _x, uint32_t _y, uint32_t _z) {
            _data = (_x << 21) + (_y << 10) + _z;
        }

        CPT_CPU_GPU_INLINE uint32_t size() const {
            return x() * y() * z();
        }

        CPT_CPU_GPU_INLINE bool is_homogeneous() const {
            return z() == 0;
        }

        explicit CPT_CPU IndexBound(uint32_t _x = 0, uint32_t _y = 0, uint32_t _z = 0):
        _data((_x << 21) + (_y << 10) + _z) {
            if (_x >= 2048 || _y >= 2048 || _z >= 1024) {
                std::cerr << "Volume too big. Max size: [2047, 2047, 1023]\n";
                throw std::runtime_error("Volume too big");
            }
        }
    };

    AABB* bounds;           // bounds of the volumes
    float* _emit_data;      // emission underlying data
    Vec4* _albedo_tag;      // underlying data of albedo and reinterpreted int tag (indicating homogeneous or heterogeneous)
    IndexBound* idx_bounds;            // offsets of different volumes
public:
    
};