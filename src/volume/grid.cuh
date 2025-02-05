#pragma once
/**
 * @file homogeneous.cuh
 * @author Qianyue He
 * @brief Grid volume
 * @version 0.1
 * @date 2025-02-05
 * @copyright Copyright (c) 2025
 */

#ifdef VALID_NANOVDB
#include "core/medium.cuh"
#include <nanovdb/NanoVDB.h>

class GridVolumeMedium: public Medium {
public:
    // GridVolumeMedium does not directly control the life-span
    // of the grid data, so it just holds the pointer to the 
    // device grid
    const nanovdb::FloatGrid* density;      // density
    const nanovdb::Vec3fGrid* albedo;       // albedo Vec3f
    const nanovdb::FloatGrid* emission;     // emission grid: blackbody temporature
    AABB grid_aabb;

    CPT_GPU GridVolumeMedium(
        const nanovdb::FloatGrid* _den,
        const nanovdb::Vec3fGrid* _alb,
        const nanovdb::FloatGrid* _em
    ): density(_den), albedo(_alb), emission(_em) {
        auto bbox = density->worldBBox();
        grid_aabb = AABB(
            Vec3(bbox.min()[0], bbox.min()[1], bbox.min()[2]),
            Vec3(bbox.max()[0], bbox.max()[1], bbox.max()[2]),
            0, 0
        );
    }
};

#endif  // VALID_NANOVDB