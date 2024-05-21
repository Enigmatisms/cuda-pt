/**
 * Emitters
 * I only plan to support spot light, area light and point source
 * 
 * @author: Qianyue He
 * @date:   5.13.2024
*/

#include "core/vec3.cuh"

CPT_CPU_GPU_INLINE float distance_attenuate(Vec3&& diff) {
    return min(1.f / max(diff.length2(), 1e-5f), 1.f);
}

class Emitter {
protected:
    Vec3 Le;
    int obj_ref_id;         // index pointing to object, -1 means the emitter is delta_pos
public:
    /**
     * Sample a point on the emitter (useful for non-infinitesimal emitters)
    */
    CPT_CPU_GPU Emitter() {}

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU Emitter(VecType&& le, int obj_ref_id = -1):
        Le(std::forward<Vec3>(le), obj_ref_id(obj_ref_id)) {}

    CPT_CPU_GPU virtual Vec3 sample(const Vec3& hit_pos, Vec3& le, float& pdf) const {
        pdf = 1;
        le.fill(0);
        return Vec3();
    }

    CPT_CPU_GPU virtual Vec3 eval_le(const Vec3* const inci_dir = nullptr, const Vec3* const normal = nullptr) const noexcept = 0;
    CPT_CPU_GPU_INLINE bool non_delta() const noexcept {
        return this->obj_ref_id >= 0;
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

    CPT_CPU_GPU Vec3 sample(const Vec3& hit_pos, Vec3& le, float& pdf) const override {
        le = this->Le * distance_attenuate(pos - hit_pos);
        pdf = 1.f;
        return this->pos;
    }

    CPT_CPU_GPU virtual Vec3 eval_le(const Vec3* const, const Vec3* const) const noexcept override {
        return Vec3(0, 0, 0);
    }
};

/**
 * TODO: Object <---> mesh relationship is not fully implemented
*/
class AreaSource: public Emitter {
protected:
    Vec3 pos;
public:
    CPT_CPU_GPU AreaSource() {}

    CONDITION_TEMPLATE_2(VType1, VType2, Vec3)
    CPT_CPU_GPU AreaSource(VType1&& le, VType2&& pos, int obj_ref_id): 
        Emitter(std::forward<VType1>(le), obj_ref_id), pos(std::forward<VType2>(pos)) {}

    CPT_CPU_GPU Vec3 sample(const Vec3& hit_pos, Vec3& le, float& pdf) const override {
        return this->pos;
    }

    CPT_CPU_GPU virtual Vec3 eval_le(const Vec3* const, const Vec3* const) const noexcept override {
        return Vec3(0, 0, 0);
    }
};