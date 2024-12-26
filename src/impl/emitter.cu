#include "core/emitter.cuh"
#include "core/textures.cuh"

CPT_GPU Vec3 EnvMapEmitter::sample(
    const Vec3& hit_pos, const Vec3& hit_n, Vec4& le, float& pdf, 
    Vec2&& uv, const PrecomputedArray& prims, const ArrayType<Vec3>& norms, int sampled_index
) const {
    Vec3 direction = delocalize_rotate(hit_n, sample_cosine_hemisphere(uv, pdf)),
         sample_pos = hit_pos + ENVMAP_DIST * direction;
    direction = rot.rotate(direction);                      // get local direction to obtain the UV coord
    float tht_y = acosf(direction.z()) * M_1_Pi,            // [0, 1]
          phi_x = (atan2f(direction.y(), direction.x()) + M_Pi) * M_1_Pi * 0.5f;        // [0, 1]
    le = Vec4( tex2D<float4>(env, phi_x, tht_y) ) * scale;
    return sample_pos;
}

CPT_GPU Vec4 EnvMapEmitter::sample_le(
    Vec3& ray_o, Vec3& ray_d, float& pdf, 
    Vec2&& uv, const PrecomputedArray& prims, const ArrayType<Vec3>& norms, int sampled_index,
    float extra_u, float extra_v
) const {
    Vec3 local_sample = sample_uniform_sphere(std::move(uv), pdf);
    float tht_y = acosf(local_sample.z()) * M_1_Pi,            // [0, 1]
          phi_x = (atan2f(local_sample.y(), local_sample.x()) + M_Pi) * M_1_Pi * 0.5f;        // [0, 1]
    local_sample = rot.inverse_rotate(local_sample);
    ray_o = ENVMAP_DIST * local_sample;
    ray_d = -local_sample;
    return Vec4( tex2D<float4>(env, phi_x, tht_y) ) * scale;
}

CPT_GPU Vec4 EnvMapEmitter::eval_le(const Vec3* const inci_dir, const Vec3* const normal) const {
    Vec3 direction = rot.rotate(*inci_dir);
    float tht_y = acosf(direction.z()) * M_1_Pi,                                        // [0, 1]
          phi_x = (atan2f(direction.y(), direction.x()) + M_Pi) * M_1_Pi * 0.5f;        // [0, 1]
    return Vec4( tex2D<float4>(env, phi_x, tht_y) ) * scale;
}