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
    int width;
    int height;
    int cache_level;
    int spp;
    int spec_constraint;
    bool gamma_correction;
    bool bidirectional;
    float caustic_scaling;

    MaxDepthParams md;

    RenderingConfig() : width(512), height(512), cache_level(4), spp(64), 
        spec_constraint(0), gamma_correction(true), bidirectional(false), caustic_scaling(1.0) {}

    static RenderingConfig from_xml(const tinyxml2::XMLElement *sensor_node);
};