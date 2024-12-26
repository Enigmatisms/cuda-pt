#pragma once
/**
 * @brief Support for CUDA texture. Online loading is not supported
 * @author Qianyue He
 * @date   2024.12.24
 */

#include "core/so3.cuh"
#include "core/vec4.cuh"
#include "core/vec2_half.cuh"
#include "core/interaction.cuh"

// TODO: we need host side Texture support

// Host side memory management tool

enum TextureType: uint8_t {
    DIFFUSE_TEX   = 0x0,
    SPECULAR_TEX  = 0x1,
    GLOSSY_TEX    = 0x2,
    NORMAL_TEX    = 0x3,
    ROUGHNESS_TEX = 0x4,
};

struct TextureInfo {
    std::string diff_path{};
    std::string spec_path{};
    std::string glos_path{};
    std::string rough_path1{};
    std::string rough_path2{};
    std::string normal_path{};
    bool is_rough_ior{false};
};

template <typename Ty>
class Texture {
private:
    // we don't need width and height, since we are using normalized coords
    Ty* _data;
    cudaTextureObject_t _obj;
public:
    TextureType ttype;
public:
    Texture(
        std::string path, 
        TextureType _ttype, 
        std::string path2 = "", 
        bool is_roughness_ior = false,
        bool is_normal_map = false
    );      // load textures from path
    void destroy();

    inline const Ty* data() const noexcept { return _data; }
    inline cudaTextureObject_t object() const noexcept { return _obj; }
};


// This should be stored as SoA
class Textures {
private:
    int num_bsdfs;
    // For the following queue: will clear and shrink_to_fit once
    // the data are transferred from host to device
    cudaTextureObject_t* tex_queue;
public:
    cudaTextureObject_t* diff_tex;      // float4 (TBA, alpha channel is never used)
    cudaTextureObject_t* spec_tex;      // float4
    cudaTextureObject_t* glos_tex;      // float4
    cudaTextureObject_t* normals;       // float4
    cudaTextureObject_t* roughness;     // float2
public:
    // return world space normal
    CPT_GPU_INLINE Vec3 eval_normal(const Interaction& it, int index) {
        const cudaTextureObject_t norm_tex = normals[index];
        auto R_w2l = rotation_fixed_anchor(it.shading_norm, false);
        Vec3 local_n = R_w2l.rotate(it.shading_norm);
        float4 pnorm = norm_tex == 0 ? make_float4(0, 0, 1, 0) : tex2D<float4>(norm_tex, it.uv_coord.x_float(), it.uv_coord.y_float());
        return R_w2l.transposed_rotate(Vec3(pnorm.x, pnorm.y, pnorm.z).normalized());
    }

    // directly return local normal
    CPT_GPU_INLINE Vec3 eval_normal_reused(const Interaction& it, int index, SO3& R_w2l) {
        R_w2l = rotation_fixed_anchor(it.shading_norm, false);
        const cudaTextureObject_t norm_tex = normals[index];
        Vec3 local_n = R_w2l.rotate(it.shading_norm);
        float4 pnorm = norm_tex == 0 ? make_float4(0, 0, 1, 0) : tex2D<float4>(norm_tex, it.uv_coord.x_float(), it.uv_coord.y_float());
        return R_w2l.transposed_rotate(Vec3(pnorm.x, pnorm.y, pnorm.z).normalized());
    }

    CPT_GPU_INLINE Vec2 eval_rough(const Vec2Half& uv, int index, const Vec2& default_r) {
        const cudaTextureObject_t rough_tex = roughness[index];
        return rough_tex == 0 ? default_r : Vec2(tex2D<float2>(rough_tex, uv.x_float(), uv.y_float()));
    }

    CPT_GPU_INLINE Vec4 eval(const cudaTextureObject_t& target, const Vec2Half& uv, const Vec4& default_v) {
           return target == 0 ? default_v : Vec4( tex2D<float4>(target, uv.x_float(), uv.y_float()) );
    }

    void init(int _num_bsdfs) {
        num_bsdfs = _num_bsdfs;
        tex_queue = new cudaTextureObject_t[_num_bsdfs * 5];
        memset(tex_queue, 0, sizeof(cudaTextureObject_t) * _num_bsdfs * 5);
        // allocate host side
        CUDA_CHECK_RETURN(cudaMalloc(&diff_tex, sizeof(cudaTextureObject_t) * _num_bsdfs * 5));
        spec_tex  = diff_tex + _num_bsdfs;
        glos_tex  = spec_tex + _num_bsdfs;
        normals   = glos_tex + _num_bsdfs;
        roughness = normals + _num_bsdfs;
    }

    void destroy() {
        CUDA_CHECK_RETURN(cudaFree(diff_tex));
        if (tex_queue != nullptr) {
            std::cerr << "Textures not deleted. Find the instantiated texture and check whether `to_gpu` is called.\n";
            delete[] tex_queue;
        }
    }

    void to_gpu() {
        // copy to GPU, yet host std::vector<Texture> still owns a copy
        // Texture class still controls when to release the resources
        CUDA_CHECK_RETURN(cudaMemcpyAsync(diff_tex, tex_queue, 5 * num_bsdfs * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        delete[] tex_queue;
        tex_queue = nullptr;
    }

    template <typename Ty>
    CPT_CPU void enqueue(const Texture<Ty>& tex, int index) {
        uint8_t ttype = tex.ttype;
        int offset = 0;
        if (ttype < TextureType::ROUGHNESS_TEX) {
            offset = int(ttype) * num_bsdfs;
        } else if (ttype == TextureType::ROUGHNESS_TEX) {
            offset = num_bsdfs * 4;
        } else {
            std::cerr << "Unknown texture type " << uint8_t(tex.ttype) << " found during enqueue.\n";
            throw std::runtime_error("Unknown texture type");
        }
        tex_queue[index + offset] = tex.object();
    }
};

extern __constant__ Textures c_textures;

bool save_image(
    const std::string& filename,
    const std::vector<unsigned char>& image_data, 
    int width, 
    int height, 
    std::string format = "png",
    const int quality = 90
);
