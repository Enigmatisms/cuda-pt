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
    }
    updated = false;
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void BSDFInfo::create_on_gpu(BSDF*& to_store) const {
    // destroy the previous BSDF
    destroy_gpu_alloc<<<1, 1>>>(&to_store);
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
        general_bsdf_filler<TranslucentBSDF>(&to_store, bsdf, 
            static_cast<BSDFFlag>(BSDFFlag::BSDF_SPECULAR | BSDFFlag::BSDF_TRANSMIT)
        );
    } else if (type == BSDFType::Plastic) {
        create_plastic_bsdf<PlasticBSDF><<<1, 1>>>(&to_store, 
            bsdf.k_d, 
            bsdf.k_s, 
            bsdf.k_g, 
            bsdf.ior(), 
            bsdf.trans_scaler(), 
            bsdf.thickness(), 
            bsdf.kd_tex_id, 
            bsdf.ex_tex_id
        );
    } else if (type == BSDFType::PlasticForward) {
            create_plastic_bsdf<PlasticForwardBSDF><<<1, 1>>>(&to_store, 
            bsdf.k_d, 
            bsdf.k_s, 
            bsdf.k_g, 
            bsdf.ior(), 
            bsdf.trans_scaler(), 
            bsdf.thickness(), 
            bsdf.kd_tex_id, 
            bsdf.ex_tex_id
        );
    } else if (type == BSDFType::GGXConductor) {
        create_metal_bsdf<<<1, 1>>>(&to_store, 
            METAL_ETA_TS[bsdf.mtype], 
            METAL_KS[bsdf.mtype], 
            bsdf.k_g, 
            bsdf.roughness_x(), 
            bsdf.roughness_y(), 
            bsdf.kd_tex_id, bsdf.ex_tex_id
        );
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}