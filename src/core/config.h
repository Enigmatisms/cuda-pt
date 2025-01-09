/**
 * Rendering configuration
 * @author: Qianyue He
 * @date:   2024.5.24
*/

#pragma once
#include <tinyxml2.h>
#include <string>
#include "core/max_depth.h"

class RenderingConfig {
public:
    int spp;
    int width;
    int height;
    int cache_level;
    int max_node_num;
    int spec_constraint;
    bool gamma_correction;
    bool bidirectional;
    float caustic_scaling;
    float bvh_overlap_w;            // [0.5, +inf), can not be less than 0.5, otherwise SAH will be downweighted

    MaxDepthParams md;

    RenderingConfig() : spp(64), width(512), height(512), cache_level(4), max_node_num(16), 
        spec_constraint(0), gamma_correction(true), bidirectional(false), caustic_scaling(1.0), bvh_overlap_w(0.75) {}

    static RenderingConfig from_xml(
        const tinyxml2::XMLElement *acc_node,
        const tinyxml2::XMLElement *render_node,
        const tinyxml2::XMLElement *sensor_node
    );
};