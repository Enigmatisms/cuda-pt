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
#include "bsdf/bsdf.cuh"
#include "core/medium.cuh"
#include "core/preset_params.cuh"
#include "core/virtual_funcs.cuh"

class MediumInfo {
  private:
    int phase_id;

  public:
    std::string name;   // medium name
    std::string p_name; // phase function name
    MediumType mtype;
    PhaseFuncType ptype;
    mutable bool updated;
    mutable bool phase_changed; // whether we have changed the phase function

    MediumInfo(int phase_id = 0)
        : phase_id(phase_id), name(""), p_name(""),
          mtype(MediumType::Homogeneous),
          ptype(PhaseFuncType::NullForward), med_param{}, updated(false),
          phase_changed(false) {}
    MediumInfo(std::string n, std::string pn, int phase_id,
               MediumType m_t = MediumType::Homogeneous,
               PhaseFuncType p_t = PhaseFuncType::NullForward)
        : phase_id(phase_id), name(n), p_name(""), mtype(m_t),
          ptype(p_t), med_param{}, updated(false), phase_changed(false) {}

    int pid() const { return phase_id; };

    struct MediumParams {
        Vec4 sigma_a; // const albedo for grid, or absorption for homo medium
        Vec4 sigma_s; // emission scale and temperature scale
        Vec4 phase;   // phase function param (at most 4 floats)
        float scale;

        MediumParams() {}
        CONDITION_TEMPLATE_SEP_3(VType1, VType2, VType3, Vec4, Vec4, Vec4)
        MediumParams(VType1 &&_sa, VType2 &&_ss, VType3 &&_ph, float scale)
            : sigma_a(std::forward<VType1>(_sa)),
              sigma_s(std::forward<VType2>(_ss)),
              phase(std::forward<VType3>(_ph)), scale(scale) {}

        float &emission_scale() { return sigma_s.x(); }
        float &temperature_scale() { return sigma_s.y(); }
        float &g() { return phase.x(); }
        float &g1() { return phase.x(); }
        float &g2() { return phase.y(); }
        float &weight() { return phase.z(); }

        inline float emission_scale() const { return sigma_s.x(); }
        inline float temperature_scale() const { return sigma_s.y(); }
        inline float g() const { return phase.x(); }
        inline float g1() const { return phase.x(); }
        inline float g2() const { return phase.y(); }
        inline float weight() const { return phase.z(); }
    } med_param;

    // this will update the GPU data, but will not create new vptr and vtable
    void copy_to_gpu(Medium *&to_store, PhaseFunction **phases) const;

    void create_on_gpu(Medium *&medium, PhaseFunction **phases);

    void clamp_phase_vals() {
        if (std::abs(med_param.phase.x()) < 1e-3) {
            med_param.phase.x() = med_param.phase.x() >= 0 ? 1e-3f : -1e-3f;
        }
        if (std::abs(med_param.phase.y()) < 1e-3) {
            med_param.phase.y() = med_param.phase.y() >= 0 ? 1e-3f : -1e-3f;
        }
    }
};

class BSDFInfo {
  public:
    std::string name;
    BSDFType type;

    struct BSDFParams {
        Vec4 k_d;
        Vec4 k_s;
        Vec4 k_g;
        Vec4 extras;
        uint8_t mtype;

        BSDFParams() {}
        CONDITION_TEMPLATE_SEP_3(VType1, VType2, VType3, Vec4, Vec4, Vec4)
        BSDFParams(VType1 &&_k_d, VType2 &&_k_s, VType3 &&_k_g)
            : k_d(std::forward<VType1>(_k_d)), k_s(std::forward<VType2>(_k_s)),
              k_g(std::forward<VType3>(_k_g)), extras(1.5, 1, 0, 0),
              mtype(MetalType::Au) {}

        CONDITION_TEMPLATE(VecType, Vec4)
        void store_ggx_params(MetalType mt, VecType &&_k_g, float rx,
                              float ry) {
            k_g = std::forward<VecType>(_k_g);
            mtype = mt;
            roughness_x() = rx;
            roughness_y() = ry;
        }

        CONDITION_TEMPLATE(VecType, Vec4)
        void store_dispersion_params(DispersionType mt, VecType &&_k_s) {
            k_s = std::forward<VecType>(_k_s);
            mtype = mt;
        }

        void store_plastic_params(float _ior, float _trans_scaler,
                                  float _thickness) {
            ior() = _ior;
            trans_scaler() = _trans_scaler;
            thickness() = _thickness;
        }

        inline float &ior() { return extras.x(); }
        inline float &trans_scaler() { return extras.y(); }
        inline float &thickness() { return extras.z(); }
        inline float &roughness_x() { return extras.x(); }
        inline float &roughness_y() { return extras.y(); }
        inline float &penetration() { return extras.w(); }

        inline float ior() const { return extras.x(); }
        inline float trans_scaler() const { return extras.y(); }
        inline float thickness() const { return extras.z(); }
        inline float roughness_x() const { return extras.x(); }
        inline float roughness_y() const { return extras.y(); }
        inline float penetration() const { return extras.w(); }
    } bsdf;
    mutable bool
        updated; // whether the parameters are updated (BSDF type not changed)
    mutable bool bsdf_changed; // whether we have changed the BSDF type
    bool in_use; // whether the current BSDF is actually in use (may only appear
                 // in xml, but not used)
  public:
    BSDFInfo()
        : name(""), type(BSDFType::Lambertian), bsdf{}, updated(false),
          bsdf_changed(false), in_use(false) {}
    BSDFInfo(std::string n, BSDFType t = BSDFType::Lambertian)
        : name(n), type(t), bsdf{}, updated(false), bsdf_changed(false),
          in_use(false) {}

    template <typename TypeBSDF>
    static void general_bsdf_filler(BSDF **to_store, const BSDFParams &data,
                                    ScatterStateFlag flag) {
        create_bsdf<TypeBSDF>
            <<<1, 1>>>(to_store, data.k_d, data.k_s, data.k_g, flag);
    }

    // clamp the k_d, k_s, k_g into [0, 1], why doing this?
    // when switching from GGXConductor to 'standard' BSDF, the
    // k_d (conductor eta) and k_s (k) can be out-of-range
    void bsdf_value_clamping();

    // this will update the GPU data, but will not create new vptr and vtable
    void copy_to_gpu(BSDF *&to_store) const;

    void create_on_gpu(BSDF *&to_store);
};
