#include <omp.h>
#include <iostream>
#include "core/textures.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "ext/stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ext/stb/stb_image_write.h"

__constant__ Textures c_textures;

static bool save_image(
    const std::string& filename,
    const std::vector<unsigned char>& image_data, 
    int width, 
    int height, 
    std::string format,
    const int quality
) {
    if (format == "png") {
        // last parameter: line size
        return stbi_write_png(filename.c_str(), width, height, 4, image_data.data(), width * 4);
    } else if (format == "jpg" || format == "jpeg") {
        // last parameter: compression quality (0 - 100)
        return stbi_write_jpg(filename.c_str(), width, height, 4, image_data.data(), quality);
    } else {
        std::cerr << "Unsupported format: " << format << std::endl;
        return false;
    }
}

static bool load_image_to_float4(
    const std::string& filename, 
    std::vector<float4>& out_data, 
    int& width, int& height, 
    float offset = 0.f, float scale = 1.f
) {
    int n_channels;
    // force 4 channels
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &n_channels, 4);
    if (!data) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return false;
    }

    int num_pixels = width * height;
    out_data.resize(num_pixels);

    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < num_pixels; ++i) {
        out_data[i].x = (static_cast<float>(data[4 * i + 0]) / 255.0f) * scale + offset;
        out_data[i].y = (static_cast<float>(data[4 * i + 1]) / 255.0f) * scale + offset;
        out_data[i].z = (static_cast<float>(data[4 * i + 2]) / 255.0f) * scale + offset;
        out_data[i].w = (static_cast<float>(data[4 * i + 3]) / 255.0f) * scale + offset;
    }

    stbi_image_free(data);
    return true;
}

/**
 * @brief load two maps and composed them to a float2 map
 * note that if the second file is not presented, the second value is set to 0.01
 * @param offset     Allow the function to offset the value
 * @param scale      Allow the function to scale the value
 * @param to_alpha   If true, roughness to alpha mapping will be applied (for roughness map)
 */
static bool load_composed_float2(
    std::string file1, 
    std::string file2, 
    std::vector<float2>& out_data, 
    int& width, int& height,
    float offset = 0.0f,
    float scale  = 1.0f,
    bool to_alpha = false
) {
    int n_channels, w2, h2;
    // force 4 channels
    unsigned char* data1 = stbi_load(file1.c_str(), &width, &height, &n_channels, 1);
    unsigned char* data2 = file2.length() > 1 ? stbi_load(file2.c_str(), &w2, &h2, &n_channels, 1) : nullptr;

    if (!data1) {
        std::cerr << "Failed to load primary image: " << file1 << std::endl;
        return false;
    }
    if (data2) {
        if (w2 != width || h2 != height) {
            std::cerr << "Composed image size mismatch: (" << width << ", " << height <<
                "), (" << w2 << ", " << h2 << ")\n"; 
            return false;
        }
    }

    int num_pixels = width * height;
    out_data.resize(num_pixels);

    float sum = 0.0;
    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < num_pixels; ++i) {
        float v1 = static_cast<float>(data1[i + 0]) / 255.0f,
              v2 = data2 ? static_cast<float>(data2[i + 1]) / 255.0f : v1;
        v1 = v1 * scale + offset;
        v2 = v2 * scale + offset;
        out_data[i].x = to_alpha ? roughness_to_alpha(v1) : v1;
        out_data[i].y = to_alpha ? roughness_to_alpha(v2) : v2;
    }


    stbi_image_free(data1);
    if (data2) stbi_image_free(data2);
    return true;
}

template <typename T>
static cudaTextureObject_t createTexture2D(const T* host_data, int width, int height, T** d_ptr_out)
{
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<T>();

    T* d_ptr;
    size_t pitch;
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_ptr, &pitch, width * sizeof(T), height));

    size_t host_pitch = width * sizeof(T);
    CUDA_CHECK_RETURN(cudaMemcpy2D(d_ptr, pitch, host_data, host_pitch, host_pitch, height, cudaMemcpyHostToDevice));

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypePitch2D;
    res_desc.res.pitch2D.devPtr = d_ptr;
    res_desc.res.pitch2D.desc = channel_desc;
    res_desc.res.pitch2D.width = width;
    res_desc.res.pitch2D.height = height;
    res_desc.res.pitch2D.pitchInBytes = pitch;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeClamp;      
    tex_desc.addressMode[1] = cudaAddressModeClamp;      
    tex_desc.filterMode = cudaFilterModeLinear;           
    tex_desc.readMode = cudaReadModeElementType;         
    tex_desc.normalizedCoords = 1;                       

    cudaTextureObject_t tex_obj = 0;
    CUDA_CHECK_RETURN(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL));

    *d_ptr_out = d_ptr;
    return tex_obj;
}

template <typename TexTy>
Texture<TexTy>::Texture(
    std::string path, 
    TextureType _ttype, 
    std::string path2, 
    bool is_roughness_ior,
    bool is_normal_map
): ttype(_ttype) {
    std::vector<TexTy> host_data;
    int width = 0, height = 0;
    bool result = false;
    // Note that we don't perform type check. For example, _ttype = NORMAL_TEX, while TexTy is float2, is allowed.
    // Allowed, but the code will break down. Sure, allowed, huh.
    if constexpr (std::is_same_v<TexTy, float4>) {
        result = load_image_to_float4(path, host_data, width, height, is_normal_map ? -1.f: 0.f, is_normal_map ? 2.f : 1.f);
    } else {
        if (is_roughness_ior) {
            result = load_composed_float2(path, path2, host_data, width, height, 1, 1.5, true);
        } else {
            result = load_composed_float2(path, path2, host_data, width, height, 0, 1, false);        // max ior: 2.5, range [1, 2.5]
        }
    }
    if (result == false) {
        std::cerr << "Texture '" << path << "' failed to load." << std::endl;
        throw std::runtime_error("Failed to load texture resources.");
    }
    _obj = createTexture2D<TexTy>(host_data.data(), width, height, &_data);
}

template <typename TexTy>
void Texture<TexTy>::destroy() {
    CUDA_CHECK_RETURN(cudaFree(_data));
    CUDA_CHECK_RETURN(cudaDestroyTextureObject(_obj));
}

template Texture<float2>;
template Texture<float4>;