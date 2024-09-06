/**
 * Rendering configuration
 * @author: Qianyue He
 * @date:   2024.5.24
*/

#pragma once
#include <tinyxml2.h>
#include <string>

class RenderingConfig {
public:
    int width;
    int height;
    int max_depth;
    int spp;
    bool gamma_correction;

    RenderingConfig() : width(512), height(512), max_depth(16), spp(64), gamma_correction(true) {}

    static RenderingConfig from_xml(const tinyxml2::XMLElement *sensor_node);
};