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
 * @brief ReSTIR DI data structures and device functions
 * @date: 2025.4.10
 */
#pragma once
#include "core/interaction.cuh"
#include "core/medium.cuh"
#include "renderer/tracing_func.cuh"

static constexpr int RESTIR_MAX_M = 64;

// ─── Data Structures ─────────────────────────────────────────────────

struct ReSTIRSample {
    Vec3 light_pos;      // 12B: sampled position (or direction for envmap)
    Vec3 light_normal;   // 12B: normal at emitter surface (0 for point/envmap)
    Vec4 Le;             // 16B: raw emission (un-attenuated for point sources,
                         //     cos-gated for area sources, envmap radiance)
    float q_area;        //  4B: area-measure PDF before solid-angle conversion
    int emitter_idx;     //  4B: c_emitter index (1-based, 0=invalid)
    int emitter_prim_id; //  4B: BVH primitive index on the emitter
    int emitter_type;    //  4B: 0=area, 1=envmap, 2=point source
};
// Total: 56 bytes

struct ReSTIRReservoir {
    ReSTIRSample sample; // 56B
    float w_sum;         //  4B: Σ luminance(h_i)
    int M;               //  4B: total candidates considered
};
// Total: 64 bytes

struct ReSTIRGBuffer {
    Vec3 hit_pos;      // 12B
    Vec3 shading_norm; // 12B
    Vec3 ray_d;        // 12B
    Vec2Half uv_coord; //  4B
    int material_id;   //  4B
    float hit_dist;    //  4B
    uchar4 vol_stack;  //  4B: BankStack copy
    int8_t is_volume;  //  1B
    int8_t is_valid;   //  1B
    int8_t pad[2];     //  2B
};
// Total: 56 bytes

// ─── Device Functions ────────────────────────────────────────────────

CPT_GPU_INLINE float restir_luminance(const Vec4 &v) {
    return 0.2126f * v.x() + 0.7152f * v.y() + 0.0722f * v.z();
}

/**
 * @brief Compute h = direct_comp * bsdf_eval / direct_pdf at a given hit
 * point for a stored sample. Used during temporal/spatial reuse to reevaluate
 * the contribution weight from a different GBuffer position.
 */
CPT_GPU_INLINE Vec4 restir_compute_h(const ReSTIRSample &s,
                                     const ReSTIRGBuffer &gb,
                                     MediumPtrArray media, int envmap_id) {
    Vec3 dir;
    float direct_pdf;
    Vec4 direct_comp;

    if (s.emitter_type == 1) {
        // EnvMap: direction is constant from any hit point.
        // EnvMapEmitter::sample() overwrites direct_pdf with the cosine
        // hemisphere PDF (cos_theta / Pi), NOT multiplying q_area.
        // So we must recompute the directional PDF at the new hit point.
        dir = s.light_pos;
        direct_comp = s.Le;
        float cos_theta = max(dir.dot(gb.shading_norm), 0.f);
        direct_pdf = cos_theta * M_1_Pi;
    } else if (s.emitter_type == 2) {
        // Point source: Le is un-attenuated, apply distance attenuation
        dir = s.light_pos - gb.hit_pos;
        float dist2 = dir.length2();
        float dist = sqrtf(dist2);
        dir *= __frcp_rn(max(dist, EPSILON));
        direct_comp = s.Le * distance_attenuate(s.light_pos - gb.hit_pos);
        direct_pdf = s.q_area; // PointSource doesn't modify PDF
    } else {
        // Area light (emitter_type == 0): recompute geometry from new hit point
        dir = s.light_pos - gb.hit_pos;
        float dist2 = dir.length2();
        float dist = sqrtf(dist2);
        dir *= __frcp_rn(max(dist, EPSILON));

        float cos_light = 1.f; // default fallback
        if (s.light_normal.length2() > EPSILON) {
            cos_light = max(s.light_normal.dot(-dir), 0.f);
            direct_comp = s.Le * float(cos_light > 0.f);
            direct_pdf = s.q_area * dist2 / max(cos_light, EPSILON);
        } else {
            // Fallback: treat as area light with unknown normal
            direct_comp = s.Le;
            direct_pdf = s.q_area * dist2;
        }
    }

    // BSDF or phase function evaluation
    if (gb.is_volume) {
        int med_idx =
            gb.vol_stack.x > 0 ? (&gb.vol_stack.x)[gb.vol_stack.x] : 0;
        float phase_val = media[med_idx]->eval(dir, gb.ray_d);
        return direct_comp * Vec4(phase_val, 1.f) / max(direct_pdf, EPSILON);
    } else {
        Interaction it(gb.shading_norm, gb.uv_coord);
        Vec4 bsdf_val =
            c_material[gb.material_id]->eval(it, dir, gb.ray_d, gb.material_id);
        return direct_comp * bsdf_val / max(direct_pdf, EPSILON);
    }
}

/** Streaming WRS: add one candidate to the reservoir */
CPT_GPU_INLINE void restir_update_reservoir(ReSTIRReservoir &r,
                                            const ReSTIRSample &s, float weight,
                                            float rand) {
    r.w_sum += weight;
    r.M++;
    if (rand * r.w_sum < weight) {
        r.sample = s;
    }
}

/** Merge two reservoirs for temporal/spatial reuse */
CPT_GPU_INLINE void restir_merge_reservoirs(ReSTIRReservoir &dst,
                                            const ReSTIRReservoir &src,
                                            float src_w_sum_adjusted,
                                            float rand) {
    float total_w = dst.w_sum + src_w_sum_adjusted;
    if (total_w <= 0.f)
        return;

    if (rand * total_w < src_w_sum_adjusted) {
        dst.sample = src.sample;
    }
    dst.w_sum = total_w;
    dst.M += src.M;
    // Clamp M to prevent unbounded growth
    if (dst.M > RESTIR_MAX_M) {
        dst.w_sum *= float(RESTIR_MAX_M) / float(dst.M);
        dst.M = RESTIR_MAX_M;
    }
}
