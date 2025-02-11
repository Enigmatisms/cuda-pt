#pragma once
/**
 * @file medium.cuh
 * @author Qianyue He
 * @brief Medium base class definition
 * @version 0.1
 * @date 2025-02-02
 * @copyright Copyright (c) 2025
 */

#include <type_traits>
#include "core/so3.cuh"
#include "core/aabb.cuh"
#include "core/phase.cuh"   

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

// POD: distance sampling sample
struct MediumSample {
    Vec4 local_thp;
    float dist;
    float pdf;
    uint32_t flag;
};

class Medium {
protected:
    PhaseFunction* phase;
public:
    CPT_CPU_GPU Medium(): phase(nullptr) {}

    CPT_CPU_GPU virtual ~Medium() {}
public:
    // distance sampling: decide whether it is medium event or surface event
    CPT_GPU virtual MediumSample sample(const Ray& ray, Sampler& sp, float max_dist = MAX_DIST) const {
        return {Vec4(1), ray.hit_t, 1, 0};      // return surface event by default
    }
    // evaluate transmittance given the ray direction and distance
    CPT_GPU virtual Vec4 transmittance(const Ray& ray, Sampler& sp, float dist) const {
        return Vec4(1);
    }

    // phase function sampling, update the ray direction
    CPT_GPU_INLINE virtual Vec3 scatter(Vec3 raydir, Vec4& throughput, Sampler& sp) const {
        PhaseSample psp = phase->sample(sp, raydir);
        throughput *= psp.weight;
        return delocalize_rotate(raydir, std::move(psp.outdir));
    }

    CPT_CPU_GPU_INLINE void bind_phase_function(PhaseFunction* ptr) {
        phase = ptr;
    }

    // evaluate local scattering phase function
    CPT_GPU_INLINE virtual float eval(Vec3 indir, Vec3 outdir) const {
        return phase->eval(std::move(indir), std::move(outdir));
    }
};

using MediumPtrArray = Medium** const __restrict__;
