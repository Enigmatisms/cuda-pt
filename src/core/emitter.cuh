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
    bool _delta_pos;
public:
    /**
     * Sample a point on the emitter (useful for non-infinitesimal emitters)
    */
    CPT_CPU_GPU Emitter() {}

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU Emitter(VecType&& le, bool delta_pos = false):
        Le(std::forward<Vec3>(le), _delta_pos(delta_pos)) {}

    CPT_CPU_GPU virtual Vec3 sample(const Vec3& hit_pos, Vec3& le, float& pdf) const = 0;

    CPT_CPU_GPU virtual Vec3 eval_le(const Vec3* const inci_dir = nullptr, const Vec3* const normal = nullptr) const noexcept = 0;
    CPT_CPU_GPU bool is_delta_pos() const noexcept {
        return this->_delta_pos;
    }
};

class PointSource: public Emitter {
protected:
    Vec3 pos;
public:
    CPT_CPU_GPU PointSource() {}

    CONDITION_TEMPLATE_2(VType1, VType2, Vec3)
    CPT_CPU_GPU PointSource(VType1&& le, VType2&& pos, bool delta_pos = false): 
        Emitter(std::forward<VType1>(le), delta_pos), pos(std::forward<VType2>(pos)) {}

    CPT_CPU_GPU Vec3 sample(const Vec3& hit_pos, Vec3& le, float& pdf) const override {
        le = this->Le * distance_attenuate(pos - hit_pos);
        pdf = 1.f;
        return this->pos;
    }

    CPT_CPU_GPU virtual Vec3 eval_le(const Vec3* const, const Vec3* const) const noexcept override {
        return Vec3(0, 0, 0);
    }
};