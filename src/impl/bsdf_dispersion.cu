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

/**
 * @author Qianyue He
 * @brief Dispersion BSDF
 * @date 2024.11.06
 */
#include "bsdf/dispersion.cuh"
#include "core/xyz.cuh"

CPT_GPU_INLINE Vec4 DispersionBSDF::wavelength_to_XYZ(float wavelength) {
    float cie_index = wavelength - DispersionBSDF::WL_MIN,
          d65_index = wavelength - DispersionBSDF::D65_MIN;
    auto xyz =
        Vec4(tex1D<float4>(XYZ.CIE, cie_index / DispersionBSDF::WL_RANGE));
    float SPD = tex1D<float>(XYZ.D65, d65_index / DispersionBSDF::D65_RANGE);
    // Average intensity of the D65 illuminant over its wavelengths
    xyz *= SPD / 22.2175f;
    return xyz;
}

CPT_GPU_INLINE Vec4 DispersionBSDF::wavelength_to_RGB(float wavelength) {
    constexpr Vec4 scale(1.4979, 1.13591, 1.13159);
    Vec4 RGB = ColorSpaceXYZ::XYZ_to_sRGB(wavelength_to_XYZ(wavelength));
    RGB = RGB.maximize(Vec4(0));
    return RGB / scale;
}

CPT_GPU float DispersionBSDF::pdf(const Interaction &it, const Vec3 &out,
                                  const Vec3 &incid, int index) const {
    const Vec3 normal = c_textures.eval_normal(it, index);
    bool in_pos = normal.dot(incid) > 0, out_pos = normal.dot(out) > 0;
    float out_pdf = 0;
    if ((in_pos ^ out_pos) == false) { // refraction
        float wavelength = 0;
        out_pdf = get_wavelength_from(incid, out, normal, wavelength);
        float eta = get_ior(wavelength), cos_theta_i = incid.dot(normal),
              F = FresnelTerms::fresnel_simple(
                  eta, -cos_theta_i); // F is the reflected part
        out_pdf *= (1.f - F) / DispersionBSDF::WL_RANGE;
    }
    return out_pdf;
}

CPT_GPU Vec4 DispersionBSDF::eval(const Interaction &it, const Vec3 &out,
                                  const Vec3 &in, int index, bool is_mi,
                                  bool is_radiance) const {
    const Vec3 normal = c_textures.eval_normal(it, index);
    float wavelength = 0;
    Vec4 result(0, 1);
    bool valid = get_wavelength_from(in, out, normal, wavelength);
    const cudaTextureObject_t spec_tex = c_textures.spec_tex[index];
    const Vec4 ks = c_textures.eval(spec_tex, it.uv_coord, k_s);
    float eta = valid ? get_ior(wavelength) : k_d.x();
    result = TranslucentBSDF::eval_impl(normal, out, in, ks, eta, is_radiance);
    result *= valid ? wavelength_to_RGB(wavelength) : Vec4(1);
    return result;
}

CPT_GPU Vec3 DispersionBSDF::sample_dir(const Vec3 &indir,
                                        const Interaction &it, Vec4 &throughput,
                                        float &pdf, Sampler &sp,
                                        ScatterStateFlag &samp_lobe, int index,
                                        bool is_radiance) const {
    float wavelength = DispersionBSDF::sample_wavelength(sp);
    float eta = get_ior(wavelength);

    const Vec3 normal = c_textures.eval_normal(it, index);
    const cudaTextureObject_t spec_tex = c_textures.spec_tex[index];
    const Vec4 ks = c_textures.eval(spec_tex, it.uv_coord, k_s);
    auto result = TranslucentBSDF::sample_dir_impl(
        indir, normal, ks, eta, throughput, sp, pdf, samp_lobe, is_radiance);
    auto rgb = wavelength_to_RGB(wavelength);
    throughput *= rgb;
    pdf *= 1.f / DispersionBSDF::WL_RANGE;
    return result;
}
