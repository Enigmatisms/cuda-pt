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

#include "core/dynamic_bsdf.cuh"
#include "volume/medium_registry.cuh"
#include "volume/phase_registry.cuh"

void BSDFInfo::copy_to_gpu(BSDF *&to_store) const {
    if (updated == false)
        return;
    if (type == BSDFType::Lambertian) {
        load_bsdf<LambertianBSDF>
            <<<1, 1>>>(&to_store, bsdf.k_d, bsdf.k_s, bsdf.k_g);
    } else if (type == BSDFType::Specular) {
        load_bsdf<SpecularBSDF>
            <<<1, 1>>>(&to_store, bsdf.k_d, bsdf.k_s, bsdf.k_g);
    } else if (type == BSDFType::Translucent) {
        load_bsdf<TranslucentBSDF>
            <<<1, 1>>>(&to_store, bsdf.k_d, bsdf.k_s, bsdf.k_g);
    } else if (type == BSDFType::Plastic) {
        load_plastic_bsdf<PlasticBSDF><<<1, 1>>>(
            &to_store, bsdf.k_d, bsdf.k_s, bsdf.k_g, bsdf.ior(),
            bsdf.trans_scaler(), bsdf.thickness(), bsdf.penetration());
    } else if (type == BSDFType::PlasticForward) {
        load_plastic_bsdf<PlasticForwardBSDF>
            <<<1, 1>>>(&to_store, bsdf.k_d, bsdf.k_s, bsdf.k_g, bsdf.ior(),
                       bsdf.trans_scaler(), bsdf.thickness());
    } else if (type == BSDFType::GGXConductor) {
        load_metal_bsdf<<<1, 1>>>(&to_store, METAL_ETA_TS[bsdf.mtype],
                                  METAL_KS[bsdf.mtype], bsdf.k_g,
                                  bsdf.roughness_x(), bsdf.roughness_y());
    } else if (type == BSDFType::Dispersion) {
        Vec2 dis_params = DISPERSION_PARAMS[std::min(
            bsdf.mtype, (uint8_t)DispersionType::NumDispersionType)];
        load_dispersion_bsdf<<<1, 1>>>(&to_store, bsdf.k_s, dis_params.x(),
                                       dis_params.y());
    }
    updated = false;
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void BSDFInfo::bsdf_value_clamping() {
    bsdf.k_d = bsdf.k_d.maximize(Vec4(0, 1)).minimize(Vec4(1));
    bsdf.k_s = bsdf.k_s.maximize(Vec4(0, 1)).minimize(Vec4(1));
    bsdf.k_g = bsdf.k_g.maximize(Vec4(0, 1)).minimize(Vec4(1));
}

void BSDFInfo::create_on_gpu(BSDF *&to_store) {
    // destroy the previous BSDF
    bsdf_changed = false;
    // create on GPU (ensure vptr and vtables are created on GPU)
    if (type == BSDFType::Lambertian) {
        general_bsdf_filler<LambertianBSDF>(
            &to_store, bsdf,
            static_cast<ScatterStateFlag>(ScatterStateFlag::BSDF_DIFFUSE |
                                          ScatterStateFlag::BSDF_REFLECT));
    } else if (type == BSDFType::Specular) {
        general_bsdf_filler<SpecularBSDF>(
            &to_store, bsdf,
            static_cast<ScatterStateFlag>(ScatterStateFlag::BSDF_SPECULAR |
                                          ScatterStateFlag::BSDF_REFLECT));
    } else if (type == BSDFType::Translucent) {
        bsdf.k_d = Vec4(1.5f, 1); // override IoR
        general_bsdf_filler<TranslucentBSDF>(
            &to_store, bsdf,
            static_cast<ScatterStateFlag>(ScatterStateFlag::BSDF_SPECULAR |
                                          ScatterStateFlag::BSDF_TRANSMIT));
    } else if (type == BSDFType::Plastic) {
        create_plastic_bsdf<PlasticBSDF>
            <<<1, 1>>>(&to_store, bsdf.k_d, bsdf.k_s, bsdf.k_g, 1.5, 1.f, 0.5f);
    } else if (type == BSDFType::PlasticForward) {
        create_plastic_bsdf<PlasticForwardBSDF>
            <<<1, 1>>>(&to_store, bsdf.k_d, bsdf.k_s, bsdf.k_g, 1.5, 1.f, 0.5f);
    } else if (type == BSDFType::GGXConductor) {
        create_metal_bsdf<<<1, 1>>>(&to_store, METAL_ETA_TS[bsdf.mtype],
                                    METAL_KS[bsdf.mtype], bsdf.k_g, 0.003f,
                                    0.003f);
    } else if (type == BSDFType::Dispersion) {
        Vec2 dis_params = DISPERSION_PARAMS[std::min(
            bsdf.mtype, (uint8_t)DispersionType::NumDispersionType)];
        create_dispersion_bsdf<<<1, 1>>>(&to_store, bsdf.k_s, dis_params.x(),
                                         dis_params.y());
    } else if (type == BSDFType::Forward) {
        create_forward_bsdf<<<1, 1>>>(&to_store);
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void MediumInfo::copy_to_gpu(Medium *&to_store, PhaseFunction **phases) const {
    if (updated == false)
        return;
    if (mtype == MediumType::Homogeneous) {
        load_homogeneous_kernel<<<1, 1>>>(&to_store, med_param.sigma_a,
                                          med_param.sigma_s, med_param.scale);
    } else if (mtype == MediumType::Grid) {
        load_grid_kernel<<<1, 1>>>(
            &to_store, med_param.sigma_a, med_param.scale,
            med_param.temperature_scale(), med_param.emission_scale());
    }
    load_phase_kernel<<<1, 1>>>(phases, phase_id, med_param.phase);
    updated = false;
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void MediumInfo::create_on_gpu(Medium *&medium, PhaseFunction **phases) {
    // destroy the previous BSDF
    phase_changed = false;
    // create on GPU (ensure vptr and vtables are created on GPU)
    if (ptype == PhaseFuncType::Isotropic) {
        create_device_phase<IsotropicPhase><<<1, 1>>>(phases, phase_id);
    } else if (ptype == PhaseFuncType::HenyeyGreenstein) {
        create_device_phase<HenyeyGreensteinPhase>
            <<<1, 1>>>(phases, phase_id, med_param.g());
    } else if (ptype == PhaseFuncType::DuoHG) {
        create_device_phase<MixedHGPhaseFunction>
            <<<1, 1>>>(phases, phase_id, med_param.g1(), med_param.g2(),
                       med_param.weight());
    } else if (ptype == PhaseFuncType::Rayleigh) {
        create_device_phase<RayleighPhase><<<1, 1>>>(phases, phase_id);
    } else if (ptype == PhaseFuncType::SGGX) {
        // not currently supported
    }
    bind_phase_func_kernel<<<1, 1>>>(&medium, phases, phase_id);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}
