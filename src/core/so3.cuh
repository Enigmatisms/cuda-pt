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
template <typename Ty>
class SO3 {
private:
    Vec3<Ty> rows[3];

    template <typename SO3Type>
    CPT_CPU_GPU SO3<Ty> operator+(SO3Type&& so3) const {
        return SO3<Ty>(
            rows[0] + so3[0],
            rows[1] + so3[1],
            rows[2] + so3[2]
        );
    }

    template <typename SO3Type>
    CPT_CPU_GPU SO3<Ty>& operator+=(SO3Type&& so3) {
        rows[0] += so3[0];
        rows[1] += so3[1];
        rows[2] += so3[2];
        return *this;
    }

    CPT_CPU_GPU SO3<Ty> scale(Ty val) const && {
        return SO3<Ty>(
            rows[0] * val,
            rows[1] * val,
            rows[2] * val
        );
    }
public:
    CPT_CPU_GPU SO3() {
        rows[0] = Vec3<Ty>();
        rows[1] = Vec3<Ty>();
        rows[2] = Vec3<Ty>();
    }

    template <typename V1, typename V2, typename V3>
    CPT_CPU_GPU SO3(V1&& v1, V2&& v2, V3&& v3, bool row_wise = true) {
        if (row_wise) {
            rows[0] = std::forward<V1>(v1);
            rows[1] = std::forward<V1>(v2);
            rows[2] = std::forward<V1>(v3);
        } else {
            rows[0] = Vec3<Ty>(v1.x, v2.x, v3.x);
            rows[1] = Vec3<Ty>(v1.y, v2.y, v3.y);
            rows[2] = Vec3<Ty>(v1.z, v2.z, v3.z);
        }
    }

    CPT_CPU_GPU SO3(Ty val) {
        rows[0] = Vec3<Ty>(val, val, val);
        rows[1] = Vec3<Ty>(val, val, val);
        rows[2] = Vec3<Ty>(val, val, val);
    }

    CPT_CPU_GPU Vec3<Ty>& operator[](size_t index) { return rows[index]; }
    CPT_CPU_GPU const Vec3<Ty>& operator[](size_t index) const { return rows[index]; }
public:
    template <typename VecType>
    CPT_CPU_GPU Vec3<Ty> rotate(VecType&& v) const noexcept {
        return Vec3<Ty>(
            rows[0].dot(std::forward<VecType>(v)), 
            rows[1].dot(std::forward<VecType>(v)), 
            rows[2].dot(std::forward<VecType>(v))
        );
    }

    CPT_CPU_GPU SO3<Ty> T() const noexcept {
        return SO3<Ty>(rows[0], rows[1], rows[2], false);
    }

    CPT_CPU_GPU static SO3<Ty> diag(Ty val) {
        return SO3<Ty>(
            Vec3<Ty>(val, 0, 0),
            Vec3<Ty>(0, val, 0),
            Vec3<Ty>(0, 0, val)
        );
    }

    Ty determinant() const {
        return rows[0].x * (rows[1].y * rows[2].z - rows[1].z * rows[2].y) - \
               rows[0].y * (rows[1].x * rows[2].z - rows[1].z * rows[2].x) + \
               rows[0].z * (rows[1].x * rows[2].y - rows[1].y * rows[2].x);
    }

    /**
     * Do not use this for rotation matrix
     * for rotation matrix inverse, please use .T()
    */
    CPT_CPU_GPU SO3<Ty> inverse() const { 
        SO3<Ty> inv_R{};
        Ty inv_det = Ty(1) / determinant(rows[0], rows[1], rows[2]);
        inv_R[0].x = (rows[1].y * rows[2].z - rows[1].z * rows[2].y) * inv_det;
        inv_R[0].y = (rows[0].z * rows[2].y - rows[0].y * rows[2].z) * inv_det;
        inv_R[0].z = (rows[0].y * rows[1].z - rows[0].z * rows[1].y) * inv_det;
        inv_R[1].x = (rows[1].z * rows[2].x - rows[1].x * rows[2].z) * inv_det;
        inv_R[1].y = (rows[0].x * rows[2].z - rows[0].z * rows[2].x) * inv_det;
        inv_R[1].z = (rows[0].z * rows[1].x - rows[0].x * rows[1].z) * inv_det;
        inv_R[2].x = (rows[1].x * rows[2].y - rows[1].y * rows[2].x) * inv_det;
        inv_R[2].y = (rows[0].y * rows[2].x - rows[0].x * rows[2].y) * inv_det;
        inv_R[2].z = (rows[0].x * rows[1].y - rows[0].y * rows[1].x) * inv_det;
    }

    template <typename U>
    friend CPT_CPU_GPU SO3<U> rotation_between(const Vec3<U>& from, const Vec3<U>& to);
};

template <typename Ty>
CPT_CPU_GPU SO3<Ty> skew_symmetry(const Vec3<Ty>& v) {
    return SO3<Ty>(
        Vec3<Ty>(0, -v.z, v.y),
        Vec3<Ty>(v.z, 0, -v.x),
        Vec3<Ty>(-v.y, v.x, 0)
    );
}

template <typename Ty>
CPT_CPU_GPU SO3<Ty> vec_mul(const Vec3<Ty>& v1, const Vec3<Ty>& v2) {
    return SO3<Ty>(
        v2 * v1.x,
        v2 * v1.y,
        v2 * v1.z
    );
}

template <typename Ty>
CPT_CPU_GPU SO3<Ty> rotation_between(const Vec3<Ty>& from, const Vec3<Ty>& to) {
    auto axis = from.cross(to);
    Ty cos_theta = from.dot(to);
    SO3<Ty> R{};
    R = SO3<Ty>::diag(cos_theta);
    if (abs(cos_theta) < Ty(1) - Ty(1e-5)) {
        auto skew = skew_symmetry(axis);
        auto norm_axis = axis.normalized();
        R += vec_mul(norm_axis, norm_axis).scale(Ty(1) - cos_theta) + skew_symmetry(axis);
    }
    return R;
}
