/**
 * I have to write the linear algebra myself...
 * this is the definition of rotation matrix
 * @date: 4.29.2024
 * @author: Qianyue He
*/
#pragma once
#include <iostream>
#include "core/vec3.cuh"
#include "core/vec4.cuh"
#include "core/quaternion.cuh"

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
    CPT_CPU_GPU SO3(V1&& v1, V2&& v2, V3&& v3) {
        rows[0] = std::forward<V1>(v1);
        rows[1] = std::forward<V2>(v2);
        rows[2] = std::forward<V3>(v3);
    }

    template <typename V1, typename V2, typename V3>
    CPT_CPU_GPU SO3(V1&& v1, V2&& v2, V3&& v3, bool row_wise) {
        // overload, this function will call column wise construction
        rows[0] = Vec3(v1.x(), v2.x(), v3.x());
        rows[1] = Vec3(v1.y(), v2.y(), v3.y());
        rows[2] = Vec3(v1.z(), v2.z(), v3.z());
    }

    CPT_CPU_GPU SO3(float val) {
        rows[0] = Vec3(val, val, val);
        rows[1] = Vec3(val, val, val);
        rows[2] = Vec3(val, val, val);
    }

    CPT_CPU_GPU Vec3& operator[](size_t index) { return rows[index]; }
    CPT_CPU_GPU const Vec3& operator[](size_t index) const { return rows[index]; }
public:

    // R @ v
    CONDITION_TEMPLATE(VecType, Vec3)
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

    CPT_CPU_GPU_INLINE Vec3 col(int index) const {
        return Vec3(rows[0][index], rows[1][index], rows[2][index]);
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

    CPT_CPU_GPU Vec3 inverse_transform_precomputed(Vec3&& v, float a20, float a21, float a22) const { 
        // use 3 vectors to avoid register scoreboard
        Vec3 temp1, temp2;
        temp1[0] = fmaf(rows[1].y(), rows[2].z(), - rows[1].z() * rows[2].y());
        temp1[1] = fmaf(rows[0].z(), rows[2].y(), - rows[0].y() * rows[2].z());
        temp1[2] = fmaf(rows[0].y(), rows[1].z(), - rows[0].z() * rows[1].y());

        temp2[0] = fmaf(rows[1].z(), rows[2].x(), - rows[1].x() * rows[2].z());
        temp2[1] = fmaf(rows[0].x(), rows[2].z(), - rows[0].z() * rows[2].x());
        temp2[2] = fmaf(rows[0].z(), rows[1].x(), - rows[0].x() * rows[1].z());
        Vec3 temp3(temp1.x(), temp2.x(), a20);
        // other elements can not be precomputed, due to 
        float inv_det = 1.f / temp3.dot(rows[0]);
        return Vec3(temp1.dot(v) * inv_det, temp2.dot(v) * inv_det, (a20 * v.x() + a21 * v.y() + a22 * v.z()) * inv_det);
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU Vec3 transposed_rotate(VecType&& v) const { 
        return Vec3(
            rows[0].x() * v.x() + rows[1].x() * v.y() + rows[2].x() * v.z(),
            rows[0].y() * v.x() + rows[1].y() * v.y() + rows[2].y() * v.z(),
            rows[0].z() * v.x() + rows[1].z() * v.y() + rows[2].z() * v.z()
        );
    }

    CPT_CPU_GPU SO3 operator*(const SO3& R2) {
        SO3 output;
        for (int i = 0; i < 3; i++) {
            auto row = rows[i];
            output[i] = Vec3(
                row.x() * R2[0][0] + row.y() * R2[1][0] + row.z() * R2[2][0],
                row.x() * R2[0][1] + row.y() * R2[1][1] + row.z() * R2[2][1],
                row.x() * R2[0][2] + row.y() * R2[1][2] + row.z() * R2[2][2]
            );
        }
        return output;
    }

    // to rotation matrix
    CONDITION_TEMPLATE(QuatType, Quaternion)
    CPT_CPU_GPU static SO3 from_quat(QuatType&& quat) {
        float xx = quat.x * quat.x;
        float yy = quat.y * quat.y;
        float zz = quat.z * quat.z;
        float xy = quat.x * quat.y;
        float xz = quat.x * quat.z;
        float yz = quat.y * quat.z;
        float wx = quat.w * quat.x;
        float wy = quat.w * quat.y;
        float wz = quat.w * quat.z;
        SO3 output(
            Vec3(
                1.0f - 2.0f * (yy + zz),
                2.0f * (xy - wz),
                2.0f * (xz + wy)
            ),
            Vec3(
                2.0f * (xy + wz),
                1.0f - 2.0f * (xx + zz),
                2.0f * (yz - wx)
            ),
            Vec3(
                2.0f * (xz - wy),
                2.0f * (yz + wx),
                1.0f - 2.0f * (xx + yy)
            )
        );
        return output;
    }

    friend CPT_GPU SO3 rotation_between(Vec3&& from, const Vec3& to);

    template <typename VecType>
    friend CPT_GPU SO3 rotation_fixed_anchor(VecType&& to, bool l2w = true);
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
CPT_GPU_INLINE SO3 rotation_between(Vec3&& from, const Vec3& to) {
    auto axis = from.cross(to);
    float cos_theta = from.dot(to);
    SO3 R = SO3::diag(cos_theta);
    if (abs(cos_theta) < 1.f - 1e-5f) {
        auto skew = skew_symmetry(axis);
        axis.normalize();
        R += vec_mul(axis, axis).scale(1.f - cos_theta) + skew;
    }
    return R;
}

// @param l2w: local to world? true by default
// if l2w is true, `to` will be dst vector to be transformed to
// other wise, `to` is actually the `from` vector (from world to local, local is (0, 0, 1))
template <typename VecType>
CPT_GPU_INLINE SO3 rotation_fixed_anchor(VecType&& to, bool l2w) {
    auto axis = Vec3(l2w? -to.y() : to.y(), l2w? to.x() : -to.x(), 0.f);
    SO3 R = SO3::diag(to.z());
    if (abs(to.z()) < 1.f - 1e-5f) {
        auto skew = skew_symmetry(axis);
        axis.normalize();
        R += vec_mul(axis, axis).scale(1.f - to.z()) + skew;
    }
    return R;
}

CONDITION_TEMPLATE(VecType, Vec3)
CPT_GPU_INLINE Vec3 delocalize_rotate(VecType&& anchor, const Vec3& to, const Vec3& input) {
    SO3 R = rotation_between(std::move(anchor), to);
    return R.rotate(input);
}   

// Specialized, when the anchor is (0, 0, 1)
CONDITION_TEMPLATE_SEP_2(VType1, VType2, Vec3, Vec3)
CPT_GPU_INLINE Vec3 delocalize_rotate(VType1&& to, VType2&& input) {
    return rotation_fixed_anchor(std::forward<VType1&&>(to))
          .rotate(std::forward<VType1&&>(input));
}   