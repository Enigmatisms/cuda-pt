#pragma once
#include "core/vec2.cuh"
#include "core/ray.cuh"
#include "core/enums.cuh"

constexpr float EPSILON = 1e-6f;
constexpr float SCALER  = 0.00001f;

template<typename Ty>
struct SphereBase {
    float radius; // radius

    Vec3<Ty> position;
    Vec3<Ty> emission;
    Vec3<Ty> color;

    ReflectionType reflection_type;

    void init(const float _radius, const Vec3<Ty>& _position, const Vec3<Ty>& _emission, const Vec3<Ty>& _color,
              const ReflectionType _reflection_type) {
        radius = _radius * SCALER;
        position = _position * SCALER;
        emission = _emission;
        color = _color;
        reflection_type = _reflection_type;
    }

    CPT_GPU float intersect(const Ray &r, Ty max_t = -1, Vec2<Ty>* uv = nullptr) const { // returns distance, 0 if nohit
        // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        Vec3f op = position - r.o; 
        
        float b = op.dot(r.d);
        float determinant = b * b - op.dot(op) + radius * radius;
        if (determinant < 0) {
            return 0;
        }

        determinant = sqrtf(determinant);

        if (float t = b - determinant; t > EPSILON) {
            return t;
        }

        if (float t = b + determinant; t > EPSILON) {
            return t;
        }

        return 0;
    }
};

using Sphere = SphereBase<float>;