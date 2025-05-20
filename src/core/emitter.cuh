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
 * Emitters
 * I only plan to support spot light, area light and point source
 *
 * @author: Qianyue He
 * @date:   5.13.2024
 */
#pragma once
#include "core/aos.cuh"
#include "core/interaction.cuh"
#include "core/quaternion.cuh"
#include "core/sampling.cuh"
#include "core/vec3.cuh"
#include "core/vec4.cuh"

CPT_CPU_GPU_INLINE float distance_attenuate(Vec3 &&diff) {
    return min(1.f / max(diff.length2(), 1e-5f), 1.f);
}

class Emitter {
  protected:
    Vec4 Le;
    int obj_ref_id;
    bool is_sphere; // whether the emitter binds to a sphere
  public:
    /**
     * Sample a point on the emitter (useful for non-infinitesimal emitters)
     */
    CPT_CPU_GPU Emitter() {}

    CONDITION_TEMPLATE(VecType, Vec4)
    CPT_CPU_GPU Emitter(VecType &&le, int obj_ref = -1, bool is_sphere = false)
        : Le(std::forward<VecType>(le)), obj_ref_id(obj_ref),
          is_sphere(is_sphere) {}

    //  sample_le, sample light emitted, used in light tracing, return sampled
    //  pos | dir | pdf and emission radiance
    CPT_GPU_INLINE virtual Vec4 sample_le(Vec3 &ray_o, Vec3 &ray_d, float &pdf,
                                          Vec2 &&, const PrecomputedArray &,
                                          const NormalArray &,
                                          const ConstBuffer<PackedHalf2> &, int,
                                          float _eu = 0, float _ev = 0) const {
        pdf = 1;
        ray_d = Vec3(0, 0, 1);
        return Vec4(0, 0, 0, 1);
    }

    // TODO: Refactor note: should we use passing by value instead of by
    // reference? Since sample can not be inlined
    CPT_GPU_INLINE virtual Vec3
    sample(const Vec3 &hit_pos, const Vec3 &, Vec4 &le, float &pdf, Vec2 &&,
           const PrecomputedArray &, const NormalArray &,
           const ConstBuffer<PackedHalf2> &, int) const {
        pdf = 1;
        le.fill(0);
        return Vec3(0, 0, 0);
    }

    CPT_GPU_INLINE virtual Vec4
    eval_le(const Vec3 *const inci_dir = nullptr,
            const Interaction *const it = nullptr) const {
        return Vec4(0, 0, 0, 1);
    }

    CPT_GPU_INLINE bool non_delta() const noexcept { return this->is_sphere; }

    CPT_GPU_INLINE Vec4 get_le() const noexcept { return Le; }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_GPU_INLINE void set_le(VecType &&color, float scaler) noexcept {
        Le = Vec4(color * scaler, scaler);
    }

    CPT_CPU_GPU int get_obj_ref() const noexcept {
        return obj_ref_id * (obj_ref_id >= 0);
    }

    // reserved handle, can be used to set many things, like (pos in
    // PointSource)
    CPT_GPU_INLINE virtual void set_func1(float val) noexcept {}
    CPT_GPU_INLINE virtual void set_func2(float val) noexcept {}
    CPT_GPU_INLINE virtual void set_func3(float val) noexcept {}
};

class PointSource : public Emitter {
  protected:
    Vec3 pos;

  public:
    CPT_CPU_GPU PointSource() {}

    CONDITION_TEMPLATE_SEP_2(VType1, VType2, Vec4, Vec3)
    CPT_CPU_GPU PointSource(VType1 &&le, VType2 &&pos)
        : Emitter(std::forward<VType1>(le)), pos(std::forward<VType2>(pos)) {}

    CPT_GPU_INLINE Vec3 sample(const Vec3 &hit_pos, const Vec3 &, Vec4 &le,
                               float &, Vec2 &&, const PrecomputedArray &,
                               const NormalArray &,
                               const ConstBuffer<PackedHalf2> &,
                               int) const override {
        le = this->Le * distance_attenuate(pos - hit_pos);
        return this->pos;
    }

    CPT_GPU_INLINE Vec4 sample_le(Vec3 &ray_o, Vec3 &ray_d, float &pdf,
                                  Vec2 &&uv, const PrecomputedArray &,
                                  const NormalArray &,
                                  const ConstBuffer<PackedHalf2> &, int, float,
                                  float) const override {
        ray_d = sample_uniform_sphere(uv, pdf);
        ray_o = pos;
        return Le;
    }

    CPT_GPU_INLINE virtual Vec4
    eval_le(const Vec3 *const inci_dir,
            const Interaction *const it) const override {
        return Vec4(0, 0, 0, 1);
    }
};

/**
 * TODO: Object <---> mesh relationship is not fully implemented
 */
class AreaSource : public Emitter {
  private:
    cudaTextureObject_t emission_tex;

  public:
    CPT_CPU_GPU AreaSource() {}

    CONDITION_TEMPLATE(VType, Vec4)
    CPT_CPU_GPU AreaSource(VType &&le, int obj_ref, bool is_sphere = false,
                           cudaTextureObject_t _emission_tex = 0)
        : Emitter(std::forward<VType>(le), obj_ref, is_sphere),
          emission_tex(_emission_tex) {}

    CPT_GPU_INLINE Vec4 scaled_Le(float u, float v) const {
        return Vec4(tex2D<float4>(emission_tex, u, v)) * Le.w();
    }

    CPT_GPU_INLINE Vec3 sample(const Vec3 &hit_pos, const Vec3 &hit_n, Vec4 &le,
                               float &pdf, Vec2 &&uv,
                               const PrecomputedArray &prims,
                               const NormalArray &norms,
                               const ConstBuffer<PackedHalf2> &uvs,
                               int sampled_index) const override {
        float sample_sum = uv.x() + uv.y();
        uv = select(uv, -uv + 1.f, sample_sum < 1.f);
        Vec3 sampled = uv.x() * prims.y_clipped(sampled_index) +
                       uv.y() * prims.z_clipped(sampled_index) +
                       prims.x_clipped(sampled_index);
        Vec3 normal = norms.eval(sampled_index, uv.x(), uv.y());
        Vec3 sphere_normal = sample_uniform_sphere(
            select(uv, -uv + 1.f, sample_sum < 1.f), sample_sum);
        sampled = select(sampled,
                         prims.get_sphere_point(sphere_normal, sampled_index),
                         is_sphere == false);
        normal = select(normal, sphere_normal, is_sphere == false);
        // normal needs special calculation
        sphere_normal = hit_pos - sampled;
        pdf *= sphere_normal.length2();
        sphere_normal.normalize();
        sample_sum = normal.dot(sphere_normal); // dot_light
        pdf *= float(sample_sum > 0) / sample_sum;
        auto tex_uv = uvs[sampled_index].lerp(uv.x(), uv.y()).xy_float();
        le = (emission_tex == 0 ? Le : scaled_Le(tex_uv.x, tex_uv.y)) *
             float(sample_sum > 0);
        return sampled;
    }

    CPT_GPU_INLINE Vec4 sample_le(Vec3 &ray_o, Vec3 &ray_d, float &pdf,
                                  Vec2 &&uv, const PrecomputedArray &prims,
                                  const NormalArray &norms,
                                  const ConstBuffer<PackedHalf2> &uvs,
                                  int sampled_index, float extra_u,
                                  float extra_v) const override {
        float sample_sum = uv.x() + uv.y(), pdf_dir = 1;
        uv = select(uv, -uv + 1.f, sample_sum < 1.f);
        Vec3 sampled = uv.x() * prims.y_clipped(sampled_index) +
                       uv.y() * prims.z_clipped(sampled_index) +
                       prims.x_clipped(sampled_index);
        Vec3 normal = norms.eval(sampled_index, uv.x(), uv.y());
        Vec3 sphere_normal = sample_uniform_sphere(
            select(uv, -uv + 1.f, sample_sum < 1.f), sample_sum);
        normal = select(normal, sphere_normal, is_sphere == false);
        ray_o = select(sampled,
                       prims.get_sphere_point(sphere_normal, sampled_index),
                       is_sphere == false) +
                normal * EPSILON;
        ray_d = sample_cosine_hemisphere(Vec2(extra_u, extra_v), pdf_dir);
        ray_d = delocalize_rotate(normal, ray_d);
        // input pdf is already the pdf of position (1 / area)
        pdf *= pdf_dir;
        auto tex_uv = uvs[sampled_index].lerp(uv.x(), uv.y()).xy_float();
        return (emission_tex == 0 ? Le : scaled_Le(tex_uv.x, tex_uv.y)) *
               fabsf(normal.dot(ray_d));
    }

    CPT_GPU virtual Vec4 eval_le(const Vec3 *const inci_dir,
                                 const Interaction *const it) const override {
        auto tex_uv = it->uv_coord.xy_float();
        return select(emission_tex == 0 ? Le : scaled_Le(tex_uv.x, tex_uv.y),
                      Vec4(0, 0, 0, 1), inci_dir->dot(it->shading_norm) < 0);
    }
};

// directed sources, shoots a cone
class AreaSpotSource : public Emitter {
  public:
    cudaTextureObject_t emission_tex;
    float cos_val;

  public:
    CPT_CPU_GPU AreaSpotSource() {}

    CONDITION_TEMPLATE(VType, Vec4)
    CPT_CPU_GPU AreaSpotSource(VType &&le, float _cos_val, int obj_ref,
                               bool is_sphere = false,
                               cudaTextureObject_t _emission_tex = 0)
        : Emitter(std::forward<VType>(le), obj_ref, is_sphere),
          cos_val(_cos_val), emission_tex(_emission_tex) {}

    CPT_GPU_INLINE Vec4 scaled_Le(float u, float v) const {
        return Vec4(tex2D<float4>(emission_tex, u, v)) * Le.w();
    }

    // Can reduce the code length by reusing, the code in this class comes from
    // AreaSource
    CPT_GPU_INLINE Vec3 sample(const Vec3 &hit_pos, const Vec3 &hit_n, Vec4 &le,
                               float &pdf, Vec2 &&uv,
                               const PrecomputedArray &prims,
                               const NormalArray &norms,
                               const ConstBuffer<PackedHalf2> &uvs,
                               int sampled_index) const override {
        float sample_sum = uv.x() + uv.y();
        uv = select(uv, -uv + 1.f, sample_sum < 1.f);
        Vec3 sampled = uv.x() * prims.y_clipped(sampled_index) +
                       uv.y() * prims.z_clipped(sampled_index) +
                       prims.x_clipped(sampled_index);
        Vec3 normal = norms.eval(sampled_index, uv.x(), uv.y());
        Vec3 sphere_normal = sample_uniform_sphere(
            select(uv, -uv + 1.f, sample_sum < 1.f), sample_sum);
        sampled = select(sampled,
                         prims.get_sphere_point(sphere_normal, sampled_index),
                         is_sphere == false);
        normal = select(normal, sphere_normal, is_sphere == false);
        // normal needs special calculation
        sphere_normal = hit_pos - sampled;
        pdf *= sphere_normal.length2();
        sphere_normal.normalize();
        sample_sum = normal.dot(sphere_normal); // dot_light
        pdf *= float(sample_sum > 0) / sample_sum;
        auto tex_uv = uvs[sampled_index].lerp(uv.x(), uv.y()).xy_float();
        le = (emission_tex == 0 ? Le : scaled_Le(tex_uv.x, tex_uv.y)) *
             (sample_sum > cos_val);
        return sampled;
    }

    CPT_GPU_INLINE Vec4 sample_le(Vec3 &ray_o, Vec3 &ray_d, float &pdf,
                                  Vec2 &&uv, const PrecomputedArray &prims,
                                  const NormalArray &norms,
                                  const ConstBuffer<PackedHalf2> &uvs,
                                  int sampled_index, float extra_u,
                                  float extra_v) const override {
        float sample_sum = uv.x() + uv.y(), pdf_dir = 1;
        uv = select(uv, -uv + 1.f, sample_sum < 1.f);
        Vec3 sampled = uv.x() * prims.y_clipped(sampled_index) +
                       uv.y() * prims.z_clipped(sampled_index) +
                       prims.x_clipped(sampled_index);
        Vec3 normal = norms.eval(sampled_index, uv.x(), uv.y());
        Vec3 sphere_normal = sample_uniform_sphere(
            select(uv, -uv + 1.f, sample_sum < 1.f), sample_sum);
        normal = select(normal, sphere_normal, is_sphere == false);
        ray_o = select(sampled,
                       prims.get_sphere_point(sphere_normal, sampled_index),
                       is_sphere == false) +
                normal * EPSILON;
        ray_d = sample_uniform_cone(Vec2(extra_u, extra_v), cos_val, pdf_dir);
        ray_d = delocalize_rotate(normal, ray_d);
        // input pdf is already the pdf of position (1 / area)
        pdf *= pdf_dir;
        auto tex_uv = uvs[sampled_index].lerp(uv.x(), uv.y()).xy_float();
        return (emission_tex == 0 ? Le : scaled_Le(tex_uv.x, tex_uv.y)) *
               fabsf(normal.dot(ray_d));
    }

    CPT_GPU virtual Vec4 eval_le(const Vec3 *const inci_dir,
                                 const Interaction *const it) const override {
        auto tex_uv = it->uv_coord.xy_float();
        return select(emission_tex == 0 ? Le : scaled_Le(tex_uv.x, tex_uv.y),
                      Vec4(0, 0, 0, 1),
                      inci_dir->dot(it->shading_norm) < -cos_val);
    }
};

class EnvMapEmitter : public Emitter {
  private:
    float azimuth;
    float zenith;
    float scale;
    cudaTextureObject_t env;
    Quaternion rot;

  public:
    CPT_CPU_GPU EnvMapEmitter() {}

    // Allow to rotate the HDRI env map online
    CPT_GPU EnvMapEmitter(cudaTextureObject_t _env, float _scale = 1,
                          float _azimuth = 0, float _zenith = 0)
        : Emitter(Vec4(0, 1), -1, false), env(_env), scale(_scale),
          azimuth(_azimuth), zenith(_zenith) {
        update_rot();
    }

    CPT_GPU Vec3 sample(const Vec3 &hit_pos, const Vec3 &hit_n, Vec4 &le,
                        float &pdf, Vec2 &&uv, const PrecomputedArray &prims,
                        const NormalArray &norms,
                        const ConstBuffer<PackedHalf2> &,
                        int sampled_index) const override;

    CPT_GPU Vec4 sample_le(Vec3 &ray_o, Vec3 &ray_d, float &pdf, Vec2 &&uv,
                           const PrecomputedArray &prims,
                           const NormalArray &norms,
                           const ConstBuffer<PackedHalf2> &, int sampled_index,
                           float extra_u, float extra_v) const override;

    CPT_GPU virtual Vec4 eval_le(const Vec3 *const inci_dir,
                                 const Interaction *const) const override;

    CPT_GPU_INLINE virtual void set_func1(float val) noexcept { scale = val; }
    CPT_GPU_INLINE virtual void set_func2(float val) noexcept { azimuth = val; }
    CPT_GPU_INLINE virtual void set_func3(float val) noexcept {
        zenith = val;
        update_rot(); // only call once
    }

    CPT_GPU_INLINE void update_rot() {
        auto quat_yaw = Quaternion::angleAxis(azimuth, Vec3(0, 0, 1)),
             quat_pit = Quaternion::angleAxis(zenith, Vec3(1, 0, 0));
        rot = quat_yaw * quat_pit;
    }
};
