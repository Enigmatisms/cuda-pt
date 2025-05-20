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

#include "core/emitter.cuh"
#include "core/textures.cuh"

CPT_GPU Vec3 EnvMapEmitter::sample(const Vec3 &hit_pos, const Vec3 &hit_n,
                                   Vec4 &le, float &pdf, Vec2 &&uv,
                                   const PrecomputedArray &prims,
                                   const NormalArray &norms,
                                   const ConstBuffer<PackedHalf2> &,
                                   int sampled_index) const {
    Vec3 direction;
    if (hit_n.length2() > EPSILON) {
        direction = delocalize_rotate(
            hit_n, sample_cosine_hemisphere(uv, pdf)); // surface sampling
    } else {
        direction = delocalize_rotate(
            Vec3(0, 0, 1), sample_uniform_sphere(uv, pdf)); // for medium
    }
    Vec3 sample_pos = hit_pos + ENVMAP_DIST * direction;
    direction =
        rot.rotate(direction); // get local direction to obtain the UV coord
    float tht_y = acosf(direction.z()) * M_1_Pi, // [0, 1]
        phi_x = (atan2f(direction.y(), direction.x()) + M_Pi) * M_1_Pi *
                0.5f; // [0, 1]
    le = Vec4(tex2D<float4>(env, phi_x, tht_y)) * scale;
    return sample_pos;
}

CPT_GPU Vec4 EnvMapEmitter::sample_le(Vec3 &ray_o, Vec3 &ray_d, float &pdf,
                                      Vec2 &&uv, const PrecomputedArray &prims,
                                      const NormalArray &norms,
                                      const ConstBuffer<PackedHalf2> &,
                                      int sampled_index, float extra_u,
                                      float extra_v) const {
    Vec3 local_sample = sample_uniform_sphere(std::move(uv), pdf);
    float tht_y = acosf(local_sample.z()) * M_1_Pi, // [0, 1]
        phi_x = (atan2f(local_sample.y(), local_sample.x()) + M_Pi) * M_1_Pi *
                0.5f; // [0, 1]
    local_sample = rot.inverse_rotate(local_sample);
    ray_o = ENVMAP_DIST * local_sample;
    ray_d = -local_sample;
    return Vec4(tex2D<float4>(env, phi_x, tht_y)) * scale;
}

CPT_GPU Vec4 EnvMapEmitter::eval_le(const Vec3 *const inci_dir,
                                    const Interaction *const) const {
    Vec3 direction = rot.rotate(*inci_dir);
    float tht_y = acosf(direction.z()) * M_1_Pi, // [0, 1]
        phi_x = (atan2f(direction.y(), direction.x()) + M_Pi) * M_1_Pi *
                0.5f; // [0, 1]
    return Vec4(tex2D<float4>(env, phi_x, tht_y)) * scale;
}
