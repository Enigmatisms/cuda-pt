/**
 * I have to write the linear algebra myself...
 * this is the definition of rotation matrix
 * @date: 4.29.2024
 * @author: Qianyue He
*/
#pragma once
#include <iostream>
#include "core/vec3.cuh"

/**
 * Though I say this is SO3 (orthogonal Matrix for rotation representation)
 * I have to implement some general Matrix ops, in order to satisfy the demand
*/
class SO3 {
private:
    Vec3 rows[3];

    template <typename SO3Type>
    CPT_CPU_GPU SO3 operator+(SO3Type&& so3) const {
        return SO3(
            rows[0] + so3[0],
            rows[1] + so3[1],
            rows[2] + so3[2]
        );
    }

    template <typename SO3Type>
    CPT_CPU_GPU SO3& operator+=(SO3Type&& so3) {
        rows[0] += so3[0];
        rows[1] += so3[1];
        rows[2] += so3[2];
        return *this;
    }

    CPT_CPU_GPU SO3 scale(float val) const && {
        return SO3(
            rows[0] * val,
            rows[1] * val,
            rows[2] * val
        );
    }
public:
    CPT_CPU_GPU SO3() {}

    template <typename V1, typename V2, typename V3>
    CPT_CPU_GPU SO3(V1&& v1, V2&& v2, V3&& v3, bool row_wise = true) {
        if (row_wise) {
            rows[0] = std::forward<V1>(v1);
            rows[1] = std::forward<V1>(v2);
            rows[2] = std::forward<V1>(v3);
        } else {
            rows[0] = Vec3(v1.x(), v2.x(), v3.x());
            rows[1] = Vec3(v1.y(), v2.y(), v3.y());
            rows[2] = Vec3(v1.z(), v2.z(), v3.z());
        }
    }

    CPT_CPU_GPU SO3(float val) {
        rows[0] = Vec3(val, val, val);
        rows[1] = Vec3(val, val, val);
        rows[2] = Vec3(val, val, val);
    }

    CPT_CPU_GPU Vec3& operator[](size_t index) { return rows[index]; }
    CPT_CPU_GPU const Vec3& operator[](size_t index) const { return rows[index]; }
public:
    template <typename VecType>
    CPT_CPU_GPU Vec3 rotate(VecType&& v) const noexcept {
        return Vec3(
            rows[0].dot(std::forward<VecType>(v)), 
            rows[1].dot(std::forward<VecType>(v)), 
            rows[2].dot(std::forward<VecType>(v))
        );
    }

    CPT_CPU_GPU SO3 T() const noexcept {
        return SO3(rows[0], rows[1], rows[2], false);
    }

    CPT_CPU_GPU static SO3 diag(float val) {
        return SO3(
            Vec3(val, 0, 0),
            Vec3(0, val, 0),
            Vec3(0, 0, val)
        );
    }

    CPT_CPU_GPU float determinant() const {
        return 
        rows[0].x() * fmaf(rows[1].y(), rows[2].z(), - rows[1].z() * rows[2].y()) - 
               rows[0].y() * fmaf(rows[1].x(), rows[2].z(), - rows[1].z() * rows[2].x()) + \
               rows[0].z() * fmaf(rows[1].x(), rows[2].y(), - rows[1].y() * rows[2].x());
    }

    /**
     * Do not use this for rotation matrix
     * for rotation matrix inverse, please use .T()
    */
    CPT_CPU_GPU SO3 inverse() const { 
        SO3 inv_R{};
        float inv_det = 1.f / determinant();
        inv_R[0].x() = (rows[1].y() * rows[2].z() - rows[1].z() * rows[2].y()) * inv_det;
        inv_R[0].y() = (rows[0].z() * rows[2].y() - rows[0].y() * rows[2].z()) * inv_det;
        inv_R[0].z() = (rows[0].y() * rows[1].z() - rows[0].z() * rows[1].y()) * inv_det;
        inv_R[1].x() = (rows[1].z() * rows[2].x() - rows[1].x() * rows[2].z()) * inv_det;
        inv_R[1].y() = (rows[0].x() * rows[2].z() - rows[0].z() * rows[2].x()) * inv_det;
        inv_R[1].z() = (rows[0].z() * rows[1].x() - rows[0].x() * rows[1].z()) * inv_det;
        inv_R[2].x() = (rows[1].x() * rows[2].y() - rows[1].y() * rows[2].x()) * inv_det;
        inv_R[2].y() = (rows[0].y() * rows[2].x() - rows[0].x() * rows[2].y()) * inv_det;
        inv_R[2].z() = (rows[0].x() * rows[1].y() - rows[0].y() * rows[1].x()) * inv_det;
        return inv_R;
    }

    CPT_CPU_GPU Vec3 inverse_transform(Vec3&& v) const { 
        // use 3 vectors to avoid register scoreboard
        Vec3 temp1, temp2, temp3, output;
        float inv_det = 1.f / determinant();
        temp1[0] = fmaf(rows[1].y(), rows[2].z(), - rows[1].z() * rows[2].y());
        temp1[1] = fmaf(rows[0].z(), rows[2].y(), - rows[0].y() * rows[2].z());
        temp1[2] = fmaf(rows[0].y(), rows[1].z(), - rows[0].z() * rows[1].y());

        temp2[0] = fmaf(rows[1].z(), rows[2].x(), - rows[1].x() * rows[2].z());
        temp2[1] = fmaf(rows[0].x(), rows[2].z(), - rows[0].z() * rows[2].x());
        temp2[2] = fmaf(rows[0].z(), rows[1].x(), - rows[0].x() * rows[1].z());

        temp3[0] = fmaf(rows[1].x(), rows[2].y(), - rows[1].y() * rows[2].x());
        temp3[1] = fmaf(rows[0].y(), rows[2].x(), - rows[0].x() * rows[2].y());
        temp3[2] = fmaf(rows[0].x(), rows[1].y(), - rows[0].y() * rows[1].x());
        return Vec3(temp1.dot(v) * inv_det, temp2.dot(v) * inv_det, temp3.dot(v) * inv_det);
    }

    friend CPT_CPU_GPU SO3 rotation_between(Vec3&& from, const Vec3& to);
};

CPT_CPU_GPU_INLINE SO3 skew_symmetry(const Vec3& v) {
    return SO3(
        Vec3(0, -v.z(), v.y()),
        Vec3(v.z(), 0, -v.x()),
        Vec3(-v.y(), v.x(), 0)
    );
}

CPT_CPU_GPU_INLINE SO3 vec_mul(const Vec3& v1, const Vec3& v2) {
    return SO3(
        v2 * v1.x(),
        v2 * v1.y(),
        v2 * v1.z()
    );
}

// This can be improved (maybe not, Rodrigues tranformation is already good enough)
CPT_CPU_GPU_INLINE SO3 rotation_between(Vec3&& from, const Vec3& to) {
    auto axis = from.cross(to);
    float cos_theta = from.dot(to);
    SO3 R{};
    R = SO3::diag(cos_theta);
    if (abs(cos_theta) < 1.f - 1e-5f) {
        auto skew = skew_symmetry(axis);
        auto norm_axis = axis.normalized();
        R += vec_mul(norm_axis, norm_axis).scale(1.f - cos_theta) + skew;
    }
    return R;
}

CONDITION_TEMPLATE(VecType, Vec3)
CPT_CPU_GPU_INLINE Vec3 delocalize_rotate(VecType&& anchor, const Vec3& to, const Vec3& input) {
    SO3 R = rotation_between(std::move(anchor), to);
    return R.rotate(input);
}   