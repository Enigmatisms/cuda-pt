/**
 * Rendering configuration
 * @author: Qianyue He
 * @date:   2024.5.24
*/

#pragma once
#include <tinyxml2.h>

struct RenderingConfig {
    int width;
    int height;
    int max_depth;
    int spp;

    RenderingConfig() : width(512), height(512), max_depth(16), spp(64) {}

    void from_xml(const tinyxml2::XMLElement *sensor_node) {
        const tinyxml2::XMLElement *node = sensor_node->FirstChildElement("integer");
        while (node) {
            std::string name = node->Attribute("name");
            if (name == "sample_count")
                node->QueryIntAttribute("value", &spp);
            else if (name == "max_bounce")
                node->QueryIntAttribute("value", &max_depth);
            node = node->NextSiblingElement("integer");
        }

        node = sensor_node->FirstChildElement("film");
        if (node) {
            const tinyxml2::XMLElement* film_elem = node->FirstChildElement("integer");
            while (film_elem) {
                std::string name = film_elem->Attribute("name");
                if (name == "width") {
                    film_elem->QueryIntAttribute("value", &width);
                } else if (name == "height") {
                    film_elem->QueryIntAttribute("value", &height);
                }
                film_elem = film_elem->NextSiblingElement("integer");
            }
        }
    }
};