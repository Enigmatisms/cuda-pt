#include "core/dynamic_bsdf.cuh"

void BSDFInfo::copy_to_gpu(BSDF*& to_store) const {
    if (updated == false) return;
    if (type == BSDFType::Lambertian) {
        load_bsdf<LambertianBSDF><<<1, 1>>>(&to_store, 
            bsdf.k_d, 
            bsdf.k_s, 
            bsdf.k_g
        );
    } else if (type == BSDFType::Specular) {
        load_bsdf<SpecularBSDF><<<1, 1>>>(&to_store, 
            bsdf.k_d, 
            bsdf.k_s, 
            bsdf.k_g
        );
    } else if (type == BSDFType::Translucent) {
        load_bsdf<TranslucentBSDF><<<1, 1>>>(&to_store, 
            bsdf.k_d, 
            bsdf.k_s, 
            bsdf.k_g
        );
    } else if (type == BSDFType::Plastic) {
        load_plastic_bsdf<PlasticBSDF><<<1, 1>>>(&to_store, 
            bsdf.k_d, 
            bsdf.k_s, 
            bsdf.k_g, 
            bsdf.ior(), 
            bsdf.trans_scaler(), 
            bsdf.thickness()
        );
    } else if (type == BSDFType::PlasticForward) {
        load_plastic_bsdf<PlasticForwardBSDF><<<1, 1>>>(&to_store, 
            bsdf.k_d, 
            bsdf.k_s, 
            bsdf.k_g, 
            bsdf.ior(), 
            bsdf.trans_scaler(), 
            bsdf.thickness()
        );
    } else if (type == BSDFType::GGXConductor) {
        load_metal_bsdf<<<1, 1>>>(&to_store, 
            METAL_ETA_TS[bsdf.mtype], 
            METAL_KS[bsdf.mtype], 
            bsdf.k_g, 
            bsdf.roughness_x(), 
            bsdf.roughness_y()
        );
    } else if (type == BSDFType::Dispersion) {
        Vec2 dis_params = DISPERSION_PARAMS[std::min(bsdf.mtype, 
                (uint8_t)DispersionType::NumDispersionType)];
        load_dispersion_bsdf<<<1, 1>>>(
            &to_store,
            bsdf.k_s,
            dis_params.x(),
            dis_params.y()
        );
    }
    updated = false;
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void BSDFInfo::bsdf_value_clamping() {
    bsdf.k_d = bsdf.k_d.maximize(Vec4(0, 1)).minimize(Vec4(1));
    bsdf.k_s = bsdf.k_s.maximize(Vec4(0, 1)).minimize(Vec4(1));
    bsdf.k_g = bsdf.k_g.maximize(Vec4(0, 1)).minimize(Vec4(1));
}

void BSDFInfo::create_on_gpu(BSDF*& to_store) {
    // destroy the previous BSDF
    bsdf_changed = false;
    destroy_gpu_alloc<<<1, 1>>>(&to_store);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // create on GPU (ensure vptr and vtables are created on GPU)
    if (type == BSDFType::Lambertian) {
        general_bsdf_filler<LambertianBSDF>(&to_store, bsdf, 
            static_cast<BSDFFlag>(BSDFFlag::BSDF_DIFFUSE | BSDFFlag::BSDF_REFLECT)
        );
    } else if (type == BSDFType::Specular) {
        general_bsdf_filler<SpecularBSDF>(&to_store, bsdf, 
            static_cast<BSDFFlag>(BSDFFlag::BSDF_SPECULAR | BSDFFlag::BSDF_REFLECT)
        );
    } else if (type == BSDFType::Translucent) {
        bsdf.k_d = Vec4(1.5f, 1);                   // override IoR
        general_bsdf_filler<TranslucentBSDF>(&to_store, bsdf, 
            static_cast<BSDFFlag>(BSDFFlag::BSDF_SPECULAR | BSDFFlag::BSDF_TRANSMIT)
        );
    } else if (type == BSDFType::Plastic) {
        create_plastic_bsdf<PlasticBSDF><<<1, 1>>>(&to_store, 
            bsdf.k_d, 
            bsdf.k_s, 
            bsdf.k_g, 
            1.5, 
            1.f, 
            0.5f
        );
    } else if (type == BSDFType::PlasticForward) {
        create_plastic_bsdf<PlasticForwardBSDF><<<1, 1>>>(&to_store, 
            bsdf.k_d, 
            bsdf.k_s, 
            bsdf.k_g, 
            1.5, 
            1.f, 
            0.5f
        );
    } else if (type == BSDFType::GGXConductor) {
        create_metal_bsdf<<<1, 1>>>(&to_store, 
            METAL_ETA_TS[bsdf.mtype], 
            METAL_KS[bsdf.mtype], 
            bsdf.k_g, 
            0.003f,
            0.003f
        );
    } else if (type == BSDFType::Dispersion) {
        Vec2 dis_params = DISPERSION_PARAMS[std::min(bsdf.mtype, 
                (uint8_t)DispersionType::NumDispersionType)];
        create_dispersion_bsdf<<<1, 1>>>(&to_store, 
            bsdf.k_s, 
            dis_params.x(),
            dis_params.y()
        );
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}