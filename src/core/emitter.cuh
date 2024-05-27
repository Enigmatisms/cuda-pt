/**
 * Emitters
 * I only plan to support spot light, area light and point source
 * 
 * @author: Qianyue He
 * @date:   5.13.2024
*/
#pragma once
#include "core/vec3.cuh"
#include "core/sampling.cuh"

CPT_CPU_GPU_INLINE float distance_attenuate(Vec3&& diff) {
    return min(1.f / max(diff.length2(), 1e-5f), 1.f);
}

class Emitter {
protected:
    Vec3 Le;
    int obj_ref_id;
    bool is_sphere;         // whether the emitter binds to a sphere
public:
    /**
     * Sample a point on the emitter (useful for non-infinitesimal emitters)
    */
    CPT_CPU_GPU Emitter() {}

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU Emitter(VecType&& le, int obj_ref = -1, bool is_sphere = false):
        Le(std::forward<VecType>(le)), obj_ref_id(obj_ref), is_sphere(is_sphere) {}

    CPT_GPU_INLINE virtual Vec3 sample(const Vec3& hit_pos, Vec3& le, float& pdf, Vec2&&, ConstPrimPtr, ConstPrimPtr, int) const {
        pdf = 1;
        le.fill(0);
        return Vec3(0, 0, 0);
    }

    CPT_GPU_INLINE virtual Vec3 eval_le(const Vec3* const inci_dir = nullptr, const Vec3* const normal = nullptr) const {
        return Vec3(0, 0, 0);
    }

    CPT_GPU_INLINE bool non_delta() const noexcept {
        return this->is_sphere;
    }

    CPT_GPU_INLINE Vec3 get_le() const noexcept {
        return Le;
    }

    CPT_CPU_GPU int get_obj_ref() const noexcept {
        return obj_ref_id * (obj_ref_id >= 0);
    }
};

class PointSource: public Emitter {
protected:
    Vec3 pos;
public:
    CPT_CPU_GPU PointSource() {}

    CONDITION_TEMPLATE_2(VType1, VType2, Vec3)
    CPT_CPU_GPU PointSource(VType1&& le, VType2&& pos): 
        Emitter(std::forward<VType1>(le)), pos(std::forward<VType2>(pos)) {}

    CPT_GPU_INLINE Vec3 sample(const Vec3& hit_pos, Vec3& le, float&, Vec2&&, ConstPrimPtr, ConstPrimPtr, int) const override {
        le = this->Le * distance_attenuate(pos - hit_pos);
        return this->pos;
    }

    CPT_GPU_INLINE virtual Vec3 eval_le(const Vec3* const , const Vec3* const ) const override {
        return Vec3(0, 0, 0);
    }
};

/**
 * TODO: Object <---> mesh relationship is not fully implemented
*/
class AreaSource: public Emitter {
public:
    CPT_CPU_GPU AreaSource() {}

    CONDITION_TEMPLATE(VType, Vec3)
    CPT_CPU_GPU AreaSource(VType&& le, int obj_ref, bool is_sphere = false): 
        Emitter(std::forward<VType>(le), obj_ref, is_sphere) {}

    CPT_GPU_INLINE Vec3 sample(
        const Vec3& hit_pos, Vec3& le, float& pdf, 
        Vec2&& uv, ConstPrimPtr prims, ConstPrimPtr norms, int sampled_index
    ) const override {
        float sample_sum = uv.x() + uv.y();
        uv = select(uv, -uv + 1.f, sample_sum < 1.f);
        float diff_x = 1.f - uv.x(), diff_y = 1.f - uv.y();
        Vec3 sampled = uv.x() * prims->y[sampled_index] + uv.y() * prims->z[sampled_index] - (uv.x() + uv.y() - 1) * prims->x[sampled_index];
        Vec3 normal = (norms->x[sampled_index] * diff_x * diff_y + \
            norms->y[sampled_index] * uv.x() * diff_y + \
            norms->z[sampled_index] * uv.y() * diff_x).normalized();
        Vec3 sphere_normal = sample_uniform_sphere(select(uv, -uv + 1.f, sample_sum < 1.f), sample_sum);
        sampled = select(
            sampled, sphere_normal * prims->y[sampled_index].x() + prims->x[sampled_index],
            is_sphere == false
        );
        normal = select(normal, sphere_normal, is_sphere == false);
        // normal needs special calculation
        sphere_normal = hit_pos - sampled;
        pdf *= sphere_normal.length2();
        sphere_normal.normalize();
        sample_sum = normal.dot(sphere_normal);           // dot_light
        pdf *= float(sample_sum > 0) / sample_sum;
        le = Le * float(sample_sum > 0);
        return sampled;
    }

    CPT_GPU virtual Vec3 eval_le(const Vec3* const inci_dir, const Vec3* const normal) const override {
        return select(this->Le, Vec3(0, 0, 0), inci_dir->dot(*normal) < 0);
    }
};