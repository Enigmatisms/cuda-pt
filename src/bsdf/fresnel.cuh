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

#pragma once
/**
 * @brief Fresnel Term
 * @author: Qianyue He
 * @date: 2024.12.10
 */

#include "core/vec3.cuh"
#include "core/vec4.cuh"

class FresnelTerms {
  private:
    Vec3 eta_t; // for conductor
    Vec3 k;     // for conductor
  public:
    CPT_CPU_GPU FresnelTerms() {}

    CONDITION_TEMPLATE_SEP_2(VType1, VType2, Vec3, Vec3)
    CPT_GPU FresnelTerms(VType1 &&_eta_t, VType2 &&_k)
        : eta_t(std::forward<VType1>(_eta_t)), k(std::forward<VType1>(_k)) {}

    CPT_GPU_INLINE static bool is_total_reflection(float dot_normal, float ni,
                                                   float nr) {
        return (1.f - (ni * ni) / (nr * nr) * (1.f - dot_normal * dot_normal)) <
               0.f;
    }

    CPT_GPU static Vec3 snell_refraction(const Vec3 &incid, const Vec3 &normal,
                                         float &cos_r2, float dot_n, float ni,
                                         float nr) {
        /* Refraction vector by Snell's Law, note that an extra flag will be
         * returned */
        float ratio = ni / nr;
        cos_r2 = 1.f - (ratio * ratio) *
                           (1. - dot_n * dot_n); // refraction angle cosine
        // for ni > nr situation, there will be total reflection
        // if cos_r2 <= 0.f, then return value will be Vec3(0, 0, 0)
        return (ratio * incid - ratio * dot_n * normal +
                sgn(dot_n) * sqrtf(fabsf(cos_r2)) * normal)
                   .normalized() *
               (cos_r2 > 0.f);
    }

    // Borrowed from Tungsten: Computes hemispherical integral of
    // dielectricReflectance(ior, cos(theta))*cos(theta) This is a Monte-Carlo
    // integration
    static CPT_CPU_GPU_INLINE float
    diffuse_fresnel(float ior, const int sample_cnt = 131072) {
        double diff_fresnel = 0.0;
        float fb = fresnel_simple(ior, 0.0f);
        for (int i = 1; i <= sample_cnt; i++) {
            float cos_theta2 = float(i) / sample_cnt;
            float fa = fresnel_simple(ior, min(std::sqrt(cos_theta2), 1.0f));
            diff_fresnel += double(fa + fb) * (0.5f / sample_cnt);
            fb = fa;
        }
        return float(diff_fresnel);
    }

    // simpler Fresnel, with which you don't need to calculate Snell's law
    static CPT_CPU_GPU_INLINE float fresnel_simple(float eta,
                                                   float cos_theta_i) {
        eta = cos_theta_i < 0.0f ? 1.0f / eta : eta;
        cos_theta_i = cos_theta_i < 0.0f ? -cos_theta_i : cos_theta_i;
        float sin_theta_t2 = eta * eta * (1.0f - cos_theta_i * cos_theta_i),
              cos_theta_t = 0, result = 1;
        if (sin_theta_t2 < 1.0f) {
            cos_theta_t = sqrtf(fmaxf(1.0f - sin_theta_t2, 0.0f));

            float Rs = (eta * cos_theta_i - cos_theta_t) /
                       (eta * cos_theta_i + cos_theta_t);
            float Rp = (eta * cos_theta_t - cos_theta_i) /
                       (eta * cos_theta_t + cos_theta_i);
            result = (Rs * Rs + Rp * Rp) * 0.5f;
        }
        return result;
    }

    CPT_GPU static float fresnel_dielectric(float n_in, float n_out,
                                            float cos_inc, float cos_ref) {
        /**
            Fresnel Equation for calculating specular ratio
            Since Schlick's Approximation is not clear about n1->n2, n2->n1
           (different) effects

            This Fresnel equation is for dielectric, not for conductor
        */
        float n1cos_i = n_in * cos_inc;
        float n2cos_i = n_out * cos_inc;
        float n1cos_r = n_in * cos_ref;
        float n2cos_r = n_out * cos_ref;
        float rs = (n1cos_i - n2cos_r) / (n1cos_i + n2cos_r);
        float rp = (n1cos_r - n2cos_i) / (n1cos_r + n2cos_i);
        return 0.5f * (rs * rs + rp * rp);
    }

    CPT_GPU Vec4 fresnel_conductor(float cos_theta_i) const {
        cos_theta_i = fminf(fmaxf(cos_theta_i, -1), 1);

        float cos2_theta_i = cos_theta_i * cos_theta_i;
        float sin2_theta_i = 1. - cos2_theta_i;
        Vec3 eta2 = eta_t * eta_t;
        Vec3 etak2 = k * k;

        Vec3 t0 = eta2 - etak2 - sin2_theta_i;
        Vec3 a2plusb2 = t0 * t0 + 4 * eta2 * etak2;
        a2plusb2 =
            Vec3(sqrtf(a2plusb2.x()), sqrtf(a2plusb2.y()), sqrtf(a2plusb2.z()));
        Vec3 t1 = a2plusb2 + cos2_theta_i;
        Vec3 a = 0.5f * (a2plusb2 + t0);
        a = Vec3(sqrtf(a.x()), sqrtf(a.y()), sqrtf(a.z()));

        Vec3 t2 = 2.f * cos_theta_i * a;
        Vec3 Rs = (t1 - t2) / (t1 + t2);

        Vec3 t3 = cos2_theta_i * a2plusb2 + sin2_theta_i * sin2_theta_i;
        Vec3 t4 = t2 * sin2_theta_i;
        Vec3 Rp = Rs * (t3 - t4) / (t3 + t4);

        Rp = 0.5f * (Rp + Rs);
        return Vec4(Rp.x(), Rp.y(), Rp.z(), 1);
    }
};
