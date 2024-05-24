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

enum EmitterBinding: uint8_t {
    POINT    = 0x00,
    TRIANGLE = 0x01,
    SPHERE   = 0x02
};

class Emitter {
protected:
    Vec3 Le;
    EmitterBinding obj_ref_id;         // index pointing to object, -1 means the emitter is delta_pos
public:
    /**
     * Sample a point on the emitter (useful for non-infinitesimal emitters)
    */
    CPT_CPU_GPU Emitter() {}

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU Emitter(VecType&& le, EmitterBinding obj_ref_id = POINT):
        Le(std::forward<VecType>(le)), obj_ref_id(obj_ref_id) {}

    CPT_GPU_INLINE virtual Vec3 sample(const Vec3& hit_pos, Vec3& le, float& pdf, Vec2&&, ConstPrimPtr, int) const {
        pdf = 1;
        le.fill(0);
        return Vec3(0, 0, 0);
    }

    CPT_GPU_INLINE virtual Vec3 eval_le(const Vec3* const inci_dir = nullptr, const Vec3* const normal = nullptr) const {
        return Vec3(0, 0, 0);
    }

    CPT_GPU_INLINE bool non_delta() const noexcept {
        return this->obj_ref_id >= 0;
    }

    CPT_GPU_INLINE Vec3 get_le() const noexcept {
        return Le;
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

    CPT_GPU_INLINE Vec3 sample(const Vec3& hit_pos, Vec3& le, float& pdf, Vec2&&, ConstPrimPtr, int) const override {
        le = this->Le * distance_attenuate(pos - hit_pos);
        pdf *= 1.f;
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
    CPT_CPU_GPU AreaSource(VType&& le, EmitterBinding obj_ref_id = TRIANGLE): 
        Emitter(std::forward<VType>(le), obj_ref_id) {}

    CPT_GPU_INLINE Vec3 sample(const Vec3& hit_pos, Vec3& le, float& pdf, Vec2&& uv, ConstPrimPtr prims, int sampled_index) const override {
        float sample_sum = uv.x() + uv.y();
        Vec3 sampled = uv.x() * prims->y[sampled_index] + uv.y() * prims->z[sampled_index] - (uv.x() + uv.y()) * prims->x[sampled_index];
        sampled = select(
            prims->y[sampled_index] + prims->z[sampled_index] - 2.f * prims->x[sampled_index] - sampled,
            sampled, sample_sum > 0
        );
        sampled = select(
            sampled, sample_uniform_sphere(std::move(uv), sample_sum) * prims->y[sampled_index].x() + prims->x[sampled_index],
            obj_ref_id == TRIANGLE
        );
        // normal needs special calculation
        return sampled;
    }

    CPT_GPU virtual Vec3 eval_le(const Vec3* const , const Vec3* const) const override {
        return Vec3(0, 0, 0);
    }
};