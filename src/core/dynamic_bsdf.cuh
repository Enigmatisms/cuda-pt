#pragma once
#include "core/bsdf.cuh"
#include "core/virtual_funcs.cuh"
#include "core/preset_params.cuh"

class BSDFInfo {
public:
    std::string name;
    BSDFType    type;

    struct BSDFParams {
        Vec4 k_d;
        Vec4 k_s;
        Vec4 k_g;
        Vec3 extras;
        uint8_t mtype;

        BSDFParams() {}
        CONDITION_TEMPLATE_SEP_3(VType1, VType2, VType3, Vec4, Vec4, Vec4)
        BSDFParams(VType1&& _k_d, VType2&& _k_s, VType3&& _k_g):
            k_d(std::forward<VType1&&>(_k_d)), k_s(std::forward<VType2&&>(_k_s)), 
            k_g(std::forward<VType3&&>(_k_g)), extras(1.5, 1, 0), mtype(MetalType::Au)
        {}

        CONDITION_TEMPLATE(VecType, Vec4)
        void store_ggx_params(MetalType mt, VecType&& _k_g, float rx, float ry) {
            k_g = std::forward<VecType&&>(_k_g);
            mtype = mt;
            roughness_x() = rx;
            roughness_y() = ry;
        }

        CONDITION_TEMPLATE(VecType, Vec4)
        void store_dispersion_params(DispersionType mt, VecType&& _k_s) {
            k_s = std::forward<VecType&&>(_k_s);
            mtype = mt;
        }

        void store_plastic_params(
            float _ior, 
            float _trans_scaler, 
            float _thickness
        ) {
            ior()          = _ior;
            trans_scaler() = _trans_scaler;
            thickness()    = _thickness;
        }
        
        inline float& ior() { return extras.x(); }
        inline float& trans_scaler() { return extras.y(); }
        inline float& thickness() { return extras.z(); }
        inline float& roughness_x() { return extras.x(); }
        inline float& roughness_y() { return extras.y(); }

        inline float ior() const { return extras.x(); }
        inline float trans_scaler() const { return extras.y(); }
        inline float thickness() const { return extras.z(); }
        inline float roughness_x() const { return extras.x(); }
        inline float roughness_y() const { return extras.y(); }
    } bsdf;
    mutable bool updated;           // whether the parameters are updated (BSDF type not changed)
    mutable bool bsdf_changed;      // whether we have changed the BSDF type
    bool in_use;                    // wether the current BSDF is actually in use (may only appear in xml, but not used)
public:
    BSDFInfo(): name(""), type(BSDFType::Lambertian), bsdf{}, updated(false), bsdf_changed(false), in_use(false) {}
    BSDFInfo(std::string n, BSDFType t = BSDFType::Lambertian):
        name(n), type(t), bsdf{}, updated(false), bsdf_changed(false), in_use(false) {}

    template <typename TypeBSDF>
    static void general_bsdf_filler(BSDF** to_store, const BSDFParams& data, BSDFFlag flag) {
        create_bsdf<TypeBSDF><<<1, 1>>>(to_store, 
            data.k_d, 
            data.k_s, 
            data.k_g, 
            flag
        );
    }

    // clamp the k_d, k_s, k_g into [0, 1], why doing this?
    // when switching from GGXConductor to 'standard' BSDF, the 
    // k_d (conductor eta) and k_s (k) can be out-of-range
    void bsdf_value_clamping();

    // this will update the GPU data, but will not create new vptr and vtable
    void copy_to_gpu(BSDF*& to_store) const;

    void create_on_gpu(BSDF*& to_store);
};