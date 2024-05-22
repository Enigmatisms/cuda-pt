/**
 * Simple tile-based path tracer
 * @date: 5.12.2024
 * @author: Qianyue He
*/
#pragma once
#include "core/emitter.cuh"
#include "core/object.cuh"
#include "renderer/tracer_base.cuh"

extern __constant__ Emitter* c_emitter[9];          // c_emitter[8] is a dummy emitter
extern __constant__ BSDF*    c_material[32];

using ConstObjPtr   = const ObjInfo* const;
using ConstBSDFPtr  = const BSDF* const;
using ConstIndexPtr = const int* const;

/**
 * Occlusion test, computation is done on global memory
*/
CPT_GPU bool occlusion_test(
    const Ray& ray,
    ConstObjPtr objects,
    ConstShapePtr shapes,
    ConstAABBPtr aabbs,
    const SoA3<Vec3>& verts,
    int num_objects,
    float max_dist
) {
    float aabb_tmin = 0;
    ShapeIntersectVisitor shape_visitor(verts, ray, EPSILON, max_dist - EPSILON);
    int prim_id = 0, num_prim = 0;
    for (int obj_id = 0; obj_id < num_objects; obj_id ++) {
        if (objects[obj_id].intersect(ray, aabb_tmin) && aabb_tmin < max_dist + EPSILON) {
            // ray intersects the object
            num_prim = objects[obj_id].prim_num;
            for (int _i = 0; _i < num_prim; _i ++, prim_id ++) {
                if (aabbs[prim_id].intersect(ray, aabb_tmin) && aabb_tmin < max_dist + EPSILON) {
                    shape_visitor.set_index(prim_id);
                    float dist = variant::apply_visitor(shape_visitor, shapes[prim_id]);
                    if (dist > max_dist - EPSILON) continue;
                    return false;
                }
            }
        }
    }
    return true;
}

CPT_GPU Emitter* sample_emitter(Sampler& sampler, bool& valid, float& pdf, int num, int no_sample) {
    valid = no_sample == 0 || num > 1;
    // logic: if no_sample and num > 1, means that there is one emitter than can not be sampled
    // so we can only choose from num - 1 emitters, the following computation does this (branchless)
    // if (emit_id >= no_sample && no_sample >= 0) -> we should skip one index (the no_sample), therefore + 1
    // if invalid (there is only one emitter, and we cannot sample it), return c_emitter[8]
    // if no_sample is 0x08, then the ray hits no emitter
    num -= no_sample > 0 && num > 1;
    int emit_id = (sampler.discrete1D() % num) + 1;
    emit_id += emit_id >= no_sample && no_sample > 0;
    pdf = 1.f / float(num);
    return c_emitter[emit_id * valid];
}

/**
 * @brief this version does not employ object-level culling
 * we use shared memory to accelerate rendering instead, for object-level culling
 * shared memory might not be easy to use, since the memory granularity will be
 * too difficult to control
 * 
 * @param objects   object encapsulation
 * @param prim2obj  primitive to object index mapping: which object does this primitive come from?
 * @param verts     vertices, SoA3: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param norms     normal vectors, SoA3: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param uvs       uv coordinates, SoA3: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
 * @param camera    GPU camera model (constant memory)
 * @param image     GPU image buffer
 * @param num_prims number of primitives (to be intersected with)
 * @param max_depth maximum allowed bounce
*/
__global__ static void render_pt_kernel(
    ConstObjPtr objects,
    ConstIndexPtr prim2obj,
    ConstShapePtr shapes,
    ConstAABBPtr aabbs,
    ConstPrimPtr verts,
    ConstPrimPtr norms, 
    ConstUVPtr uvs,
    DeviceImage& image,
    int num_prims,
    int num_objects,
    int num_emitter,
    int max_depth = 1/* max depth, useless for depth renderer, 1 anyway */
) {
    int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    Sampler sampler(px + py * image.w());
    // step 1: generate ray
    Ray ray = dev_cam.generate_ray(px, py, sampler);

    // step 2: bouncing around the scene until the max depth is reached
    Interaction it;

    // A matter of design choice
    // optimization: copy at most 32 prims from global memory to shared memory

    __shared__ Vec3 s_verts[3][32];         // vertex info
    // TODO: optimization: we don't need uv and norm, we just need it for the actual intersected primitive
    // this kinda resembles deferred rendering
    __shared__ AABB s_aabbs[32];            // aabb

    SoA3<Vec3> s_verts_soa(
        reinterpret_cast<Vec3*>(&s_verts[0]),
        reinterpret_cast<Vec3*>(&s_verts[1]),
        reinterpret_cast<Vec3*>(&s_verts[2]), 32
    );
    ShapeIntersectVisitor visitor(s_verts_soa, ray, 0);
    ShapeExtractVisitor extract(*verts, *norms, *uvs, ray, 0);

    Vec3 throughput(1, 1, 1), radiance(0, 0, 0);
    float emission_weight = 1.f;

    int num_copy = (num_prims + 31) / 32, min_index = -1;   // round up
    for (int b = 0; b < max_depth; b++) {
        float min_dist = MAX_DIST;
        min_index = -1;

        // ============= step 1: ray intersection =================
        for (int cp_base = 0; cp_base < num_copy; ++cp_base) {
            // memory copy to shared memory
            int cp_base_5 = cp_base << 5, cur_idx = cp_base_5 + tid, remain_prims = min(num_prims - cp_base_5, 32);
            if (tid < 32 && cur_idx < num_prims) {        // copy from gmem to smem
                s_verts[0][tid] = verts->x[cur_idx];
                s_verts[1][tid] = verts->y[cur_idx];
                s_verts[2][tid] = verts->z[cur_idx];

                s_aabbs[tid]  = aabbs[cur_idx];
            }
            __syncthreads();
            // this might not be a good solution
            min_dist = ray_intersect(ray, shapes, s_aabbs, visitor, min_index, remain_prims, cp_base_5, min_dist);

        }

        // ============= step 2: local shading for indirect bounces ================
        // currently, I assume Lambertian BSDF
        if (min_index >= 0) {
            extract.set_index(min_index);
            auto it = variant::apply_visitor(extract, shapes[min_index]);

            // ============= step 3: next event estimation ================

            // (1) randomly pick one emitter
            int object_id   = prim2obj[min_index],
                material_id = objects[object_id].bsdf_id,
                emitter_id  = objects[object_id].emitter_id;

            bool valid_sample = false, hit_emitter = objects[object_id].is_emitter();
            float direct_pdf = 1, emitter_pdf = 1.f;
            // (2) check if the ray hits an emitter
            emitter_id = hit_emitter * emitter_id;
            radiance += emission_weight * objects[object_id].is_emitter() * throughput *\
                        c_emitter[emitter_id]->eval_le(&ray.d, &it.shading_norm);

            Emitter* emitter = sample_emitter(sampler, valid_sample, emitter_pdf, num_emitter, emitter_id);
            // (3) sample a point on the emitter (we avoid sampling the hit emitter)
            Ray shadow_ray(ray.o + ray.d * min_dist, Vec3());
            Vec3 shadow_int;
            shadow_ray.d = emitter->sample(shadow_ray.o, shadow_int, direct_pdf) - shadow_ray.o;
            
            float emit_length = shadow_ray.d.length();
            shadow_ray.d *= 1.f / emit_length;              // normalized direction

            // (3) NEE scene intersection test (possible warp divergence, but... nevermind, it seems to be )
            if (valid_sample && occlusion_test(shadow_ray, objects, shapes, aabbs, *verts, num_objects, emit_length)) {
                // MIS for BSDF / light sampling, to achieve better rendering
                direct_pdf *= emitter_pdf;
                float material_pdf = c_material[material_id]->pdf(it, shadow_ray.d, ray.d);
                float mis_w = direct_pdf / (direct_pdf + 
                    material_pdf * emitter->non_delta());
                // accumulate direct component
                Vec3 direct = throughput * shadow_int * (mis_w / emitter_pdf) * \
                    c_material[material_id]->eval(it, shadow_ray.d, ray.d);
                printf("%f, %f, %f, %f, %f, %d\n", shadow_int.x(), throughput.x(), mis_w, emitter_pdf, material_pdf, material_id);
                radiance += direct;
                if (direct.length2() > 1e-6) {
                    printf("Direct Radiance: %f, %f, %f\n", direct.x(), direct.y(), direct.z());
                }
            }

            // step 4: sample a new ray direction, bounce the 
            float sample_pdf = 0;
            Vec3 local_th;
            ray.o = std::move(shadow_ray.o);
            ray.d = c_material[material_id]->sample_dir(ray.d, it, local_th, sampler.next2D(), sample_pdf);
            throughput *= local_th * (1.f /  sample_pdf);
        }


        // TODO: MIS for emitter
    }
    __syncthreads();
    image(px, py) += radiance;
    if (radiance.length2() > 1e-6) {
        printf("Radiance: %f, %f, %f\n", radiance.x(), radiance.y(), radiance.z());
    }
}

class PathTracer: public TracerBase {
using TracerBase::shapes;
using TracerBase::aabbs;
using TracerBase::verts;
using TracerBase::norms; 
using TracerBase::uvs;
using TracerBase::image;
using TracerBase::dev_image;
using TracerBase::num_prims;
using TracerBase::w;
using TracerBase::h;
private:
    ObjInfo* obj_info;
    int*    prim2obj;
    int num_objs;
    int num_emitter;
public:
    /**
     * @param shapes    shape information (for ray intersection)
     * @param verts     vertices, SoA3: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param norms     normal vectors, SoA3: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param uvs       uv coordinates, SoA3: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
     * @param camera    GPU camera model (constant memory)
     * @param image     GPU image buffer
     * 
     * @todo: initialize emitters
     * @todo: initialize objects
    */
    PathTracer(
        const std::vector<ObjInfo>& _objs,
        const std::vector<Shape>& _shapes,
        const SoA3<Vec3>& _verts,
        const SoA3<Vec3>& _norms, 
        const SoA3<Vec2>& _uvs,
        int num_emitter,
        int width, int height
    ): TracerBase(_shapes, _verts, _norms, _uvs, width, height), 
        num_objs(_objs.size()), num_emitter(num_emitter) 
    {
        CUDA_CHECK_RETURN(cudaMallocManaged(&obj_info, num_objs * sizeof(ObjInfo)));
        CUDA_CHECK_RETURN(cudaMallocManaged(&prim2obj, num_prims * sizeof(int)));

        int prim_offset = 0;
        for (int i = 0; i < num_objs; i++) {
            obj_info[i] = _objs[i];
            int prim_num = _objs[i].prim_num;
            for (int j = 0; j < prim_num; j++)
                prim2obj[prim_offset + j] = i;
            prim_offset += prim_num;
            printf("Set %d of %d starting at %d\n", _objs[i].prim_num, i, prim_offset);
        }
        // TODO: copy all the material into constant memory
    }

    ~PathTracer() {
        CUDA_CHECK_RETURN(cudaFree(obj_info));
        CUDA_CHECK_RETURN(cudaFree(prim2obj));
    }

    CPT_CPU std::vector<uint8_t> render(
        int num_iter = 64,
        int max_depth = 1/* max depth, useless for depth renderer, 1 anyway */
    ) override {
        ProfilePhase _(Prof::DepthRenderingHost);
        {
            ProfilePhase _p(Prof::DepthRenderingDevice);
            TicToc _timer("render_kernel()", num_iter);
            // TODO: stream processing
            for (int i = 0; i < num_iter; i++) {
                // for more sophisticated renderer (like path tracer), shared_memory should be used
                render_pt_kernel<<<dim3(w >> 4, h >> 4), dim3(16, 16)>>>(
                    obj_info, prim2obj, shapes, aabbs, verts, norms, uvs, 
                    *dev_image, num_prims, num_objs, num_emitter, max_depth
                ); 
                CUDA_CHECK_RETURN(cudaDeviceSynchronize());
            }
        }
        return image.export_cpu(1.f);
    }
};

