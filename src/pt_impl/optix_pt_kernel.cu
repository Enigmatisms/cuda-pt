/**
 * Megakernel Path Tracing (Implementation)
 * @date: 9.15.2024
 * @author: Qianyue He
*/
#include "renderer/optix_pt_kernel.cuh"
#include "optix/sbt.cuh"

static constexpr int RR_BOUNCE = 2;
static constexpr float RR_THRESHOLD = 0.1;

__constant__ LaunchParams params;

template <bool render_once>
CPT_KERNEL void render_optix_kernel(
    const DeviceCamera& dev_cam, 
    const PrecomputedArray verts,
    const ArrayType<Vec3> norms, 
    const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects,
    const cudaTextureObject_t obj_idxs,
    ConstIndexPtr emitter_prims,
    DeviceImage image,
    float* __restrict__ output_buffer,
    int num_emitter,
    int seed_offset,
    int max_depth,
    int accum_cnt,
    bool gamma_corr
) {
    int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;

    Sampler sampler(px + py * image.w(), seed_offset);
    // step 1: generate ray
    int min_index = -1, object_id = 0;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    Ray ray = dev_cam.generate_ray(px, py, sampler);

    Vec4 throughput(1, 1, 1), radiance(0, 0, 0);
    float emission_weight = 1.f;
    bool hit_emitter = false;

    // step 2: bouncing around the scene until the max depth is reached
    for (int b = 0; b < max_depth; b++) {
        float prim_u = 0, prim_v = 0, min_dist = MAX_DIST;
        min_index = -1;
        // ============= step 1: ray intersection =================
        {
            unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = 0;        // payloads
            optixTrace(
                params.handle,                  // Traversable Handle
                ray.o,                          // Ray Origin
                ray.d,                          // Ray Direction
                EPSILON,                        // tmin
                MAX_DIST,                       // tmax
                0.0f,                           // Ray Time
                0,                              // Visibility Mask
                OPTIX_RAY_FLAG_NONE,            // Ray Flags
                0,                              // SBT Offset (closest hit)
                sizeof(HitGroupRecord),         // SBT Stride
                0,                              // Miss SBT Index
                p0, p1, p2, p3                  // Payloads
            );
            min_dist  = __uint_as_float(p0);
            min_index = min_dist > EPSILON ? p1 : -1;   // miss shader will return for p0, if miss
            prim_u    = __uint_as_float(p2);
            prim_v    = __uint_as_float(p3);
        }
        // ============= step 2: local shading for indirect bounces ================
        if (min_index >= 0) {
            auto it = Primitive::get_interaction_optix(norms, uvs, ray.advance(min_dist), prim_u, prim_v, min_index, object_id >= 0);
            object_id = tex1Dfetch<int>(obj_idxs, min_index);
            object_id = object_id >= 0 ? object_id : -object_id - 1;        // sphere object ID is -id - 1

            // ============= step 3: next event estimation ================
            // (1) randomly pick one emitter
            int material_id = objects[object_id].bsdf_id,
                emitter_id  = objects[object_id].emitter_id;
            hit_emitter = emitter_id > 0;

            // emitter MIS
            emission_weight = emission_weight / (emission_weight + 
                    objects[object_id].solid_angle_pdf(it.shading_norm, ray.d, min_dist) * 
                    hit_emitter * (b > 0) * ray.non_delta());

            float direct_pdf = 1;       // direct_pdf is the product of light_sampling_pdf and emitter_pdf
            // (2) check if the ray hits an emitter
            Vec4 direct_comp = throughput *\
                        c_emitter[emitter_id]->eval_le(&ray.d, &it.shading_norm);
            radiance += direct_comp * emission_weight;

            const Emitter* emitter = sample_emitter(sampler, c_emitter, direct_pdf, num_emitter, emitter_id);
            // (3) sample a point on the emitter (we avoid sampling the hit emitter)
            emitter_id = objects[emitter->get_obj_ref()].sample_emitter_primitive(sampler.discrete1D(), direct_pdf);
            emitter_id = emitter_prims[emitter_id];               // extra mapping, introduced after BVH primitive reordering
            Ray shadow_ray(ray.advance(min_dist), Vec3(0, 0, 0));
            // use ray.o to avoid creating another shadow_int variable
            shadow_ray.d = emitter->sample(shadow_ray.o, direct_comp, direct_pdf, sampler.next2D(), verts, norms, emitter_id) - shadow_ray.o;
            
            float emit_len_mis = shadow_ray.d.length();
            shadow_ray.d *= __frcp_rn(emit_len_mis);              // normalized direction

            // (3) NEE scene intersection test (possible warp divergence, but... nevermind)
            if (emitter != c_emitter[0]) {
                bool is_occluded = false;
                {
                    unsigned int p0 = 0;        // payloads
                    optixTrace(
                        params.handle,                  // Traversable Handle
                        shadow_ray.o,                   // Ray Origin
                        shadow_ray.d,                   // Ray Direction
                        EPSILON,                        // tmin
                        emit_len_mis - EPSILON,         // tmax
                        0.0f,                           // Ray Time
                        0,                              // Visibility Mask
                        OPTIX_RAY_FLAG_ENFORCE_ANYHIT,  // Ray Flags
                        sizeof(HitGroupRecord),         // SBT Offset (any hit)
                        sizeof(HitGroupRecord),         // SBT Stride
                        0,                              // Miss SBT Index
                        p0
                    );
                    is_occluded = p0 > 0;
                }
                if (!is_occluded) {
                    // MIS for BSDF / light sampling, to achieve better rendering
                    // 1 / (direct + ...) is mis_weight direct_pdf / (direct_pdf + material_pdf), divided by direct_pdf
                    emit_len_mis = direct_pdf + c_material[material_id]->pdf(it, shadow_ray.d, ray.d) * emitter->non_delta();
                    radiance += throughput * direct_comp * c_material[material_id]->eval(it, shadow_ray.d, ray.d) * \
                        (float(emit_len_mis > EPSILON) * __frcp_rn(emit_len_mis < EPSILON ? 1.f : emit_len_mis));
                    // numerical guard, in case emit_len_mis is 0
                }
            }

            // step 4: sample a new ray direction, bounce the 
            BSDFFlag sampled_lobe = BSDFFlag::BSDF_NONE;
            ray.o = std::move(shadow_ray.o);
            ray.d = c_material[material_id]->sample_dir(ray.d, it, throughput, emission_weight, sampler, sampled_lobe);
            ray.set_delta((BSDFFlag::BSDF_SPECULAR & sampled_lobe) > 0);

            if (radiance.numeric_err())
                radiance.fill(0);
            
            float max_value = throughput.max_elem_3d();
            if (b >= RR_BOUNCE && max_value < RR_THRESHOLD) {
                if (sampler.next1D() > max_value || max_value < THP_EPS) break;
                throughput *= 1. / max_value;
            }
        }
    }
    if constexpr (render_once) {
        // image will be the output buffer, there will be double buffering
        auto local_v = image(px, py) + radiance;
        image(px, py) = local_v;
        local_v *= 1.f / float(accum_cnt);
        local_v = gamma_corr ? local_v.gamma_corr() : local_v;
        FLOAT4(output_buffer[(px + py * image.w()) << 2]) = float4(local_v); 
    } else {
        image(px, py) += radiance;
    }
}

template CPT_KERNEL void render_optix_kernel<true>(
    const DeviceCamera& dev_cam, 
    const PrecomputedArray verts,
    const ArrayType<Vec3> norms, 
    const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects,
    const cudaTextureObject_t obj_idxs,
    ConstIndexPtr emitter_prims,
    DeviceImage image,
    float* __restrict__ output_buffer,
    int num_emitter,
    int seed_offset,
    int max_depth,
    int accum_cnt,
    bool gamma_corr
);

template CPT_KERNEL void render_optix_kernel<false>(
    const DeviceCamera& dev_cam, 
    const PrecomputedArray verts,
    const ArrayType<Vec3> norms, 
    const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects,
    const cudaTextureObject_t obj_idxs,
    ConstIndexPtr emitter_prims,
    DeviceImage image,
    float* __restrict__ output_buffer,
    int num_emitter,
    int seed_offset,
    int max_depth,
    int accum_cnt,
    bool gamma_corr
);