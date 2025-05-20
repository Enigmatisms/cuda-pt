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
 * @brief CUDA object information
 * @date: 2024.5.20
 */
#pragma once
#include "core/aabb.cuh"
#include "core/aos.cuh"
#include <array>

// this class is used to reduce the memory bandwidth required by `objects` in
// the renderer not the actual bottleneck, but I guess, every bit counts this is
// a POD, so we can derive this from the ObjInfo
class CompactedObjInfo {
  private:
    int bsdf_emitter; // bsdf_id (higher 16 bits, signed) and emitter_id (lower
                      // 16 bits, signed)
    int prim_offset;  // offset to the start of the primitives
    int prim_num;     // number of primitives
    float inv_area;   // inverse area
  public:
    CPT_CPU_GPU CompactedObjInfo()
        : bsdf_emitter(0), prim_offset(0), prim_num(0), inv_area(0) {}
    CPT_CPU CompactedObjInfo(int b_e_id, int prim_o, int prim_n, float inv_a)
        : bsdf_emitter(b_e_id), prim_offset(prim_o), prim_num(prim_n),
          inv_area(inv_a) {}

    CPT_CPU_GPU_INLINE int sample_emitter_primitive(uint32_t sample,
                                                    float &pdf) const {
        pdf *= inv_area;
        return (sample % uint32_t(prim_num)) + prim_offset;
    }

    CPT_GPU_INLINE float solid_angle_pdf(const Vec3 &normal,
                                         const Vec3 &incid_dir,
                                         float min_depth) const {
        return inv_area * min_depth * min_depth /
               max(fabsf(incid_dir.dot(normal)), 1e-4);
    }

    CPT_GPU_INLINE void unpack(int &bsdf_id, int &emitter_id) const noexcept {
        bsdf_id = (bsdf_emitter >> 16);
        int temp = bsdf_emitter & 0xFFFF;
        emitter_id =
            (temp ^ 0x8000) -
            0x8000; // use two's complement to account for negative emitter_id
    }
};

class ObjInfo {
  private:
    AABB _aabb;

  public:
    int bsdf_id;        // index pf the current object
    int prim_offset;    // offset to the start of the primitives
    int prim_num;       // number of primitives
    float inv_area;     // inverse area
    uint8_t emitter_id; // index to the emitter, 0 means not an emitter
  public:
    CPT_CPU void setup(const std::array<std::vector<Vec3>, 3> &prims,
                       bool is_polygon = true);

    CPT_CPU_INLINE bool is_emitter() const noexcept {
        return this->emitter_id > 0;
    }
    CPT_CPU_GPU ObjInfo(int bsdf_id, int prim_off, int prim_num,
                        uint8_t emitter_id = 0)
        : _aabb(1e5, -1e5, -1, -1), bsdf_id(bsdf_id), prim_offset(prim_off),
          prim_num(prim_num), emitter_id(emitter_id) {}

    CPT_CPU void export_bound(Vec3 &mini, Vec3 &maxi) const noexcept {
        mini.minimized(_aabb.mini);
        maxi.maximized(_aabb.maxi);
    }

    CPT_CPU CompactedObjInfo export_gpu() const noexcept {
        // masking lower 16 bits, negative preserving
        int bsdf_emitter =
            (bsdf_id << 16) | (static_cast<int>(emitter_id) & 0xFFFF);
        return CompactedObjInfo(bsdf_emitter, prim_offset, prim_num, inv_area);
    }
};
