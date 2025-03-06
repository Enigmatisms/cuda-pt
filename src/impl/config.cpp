/**
 * Rendering configuration
 * @author: Qianyue He
 * @date:   2024.5.24
*/
#include <iostream>
#include <algorithm>
#include "core/config.h"

RenderingConfig RenderingConfig::from_xml(
    const tinyxml2::XMLElement *acc_node,
    const tinyxml2::XMLElement *render_node,
    const tinyxml2::XMLElement *sensor_node
) {
    RenderingConfig config;
    {   // Renderer Element Parsing
        const tinyxml2::XMLElement *node = render_node->FirstChildElement("integer");
        while (node) {
            std::string name = node->Attribute("name");
            if (name == "sample_count")
                node->QueryIntAttribute("value", &config.spp);
            else if (name == "max_bounce")
                node->QueryIntAttribute("value", &config.md.max_depth);
            else if (name == "max_diffuse")
                node->QueryIntAttribute("value", &config.md.max_diffuse);
            else if (name == "max_specular")
                node->QueryIntAttribute("value", &config.md.max_specular);
            else if (name == "max_transmit")
                node->QueryIntAttribute("value", &config.md.max_tranmit);
            else if (name == "max_volume")
                node->QueryIntAttribute("value", &config.md.max_volume);
            else if (name == "specular_constraint")
                node->QueryIntAttribute("value", &config.spec_constraint);

            node = node->NextSiblingElement("integer");
        }

        node = render_node->FirstChildElement("float");
        while (node) {
            std::string name = node->Attribute("name");
            if (name == "caustic_scaling") {
                node->QueryFloatAttribute("value", &config.caustic_scaling);
            } else if (name == "min_time") {
                node->QueryFloatAttribute("value", &config.md.min_time);
            } else if (name == "max_time") {
                node->QueryFloatAttribute("value", &config.md.max_time);
            }
            node = node->NextSiblingElement("float");
        }

        node = render_node->FirstChildElement("bool");
        while (node) {
            std::string name = node->Attribute("name");
            if (name == "bidirectional") {
                node->QueryBoolAttribute("value", &config.bidirectional);
            }
            node = node->NextSiblingElement("bool");
        }
    }

    {   // Accelerator Element Parsing
        const tinyxml2::XMLElement *node = acc_node->FirstChildElement("integer");
        while (node) {
            std::string name = node->Attribute("name");
            if (name == "cache_level") {
                int cache_level = 0;
                node->QueryIntAttribute("value", &cache_level);
                if (cache_level < 0 || cache_level > 8) {
                    std::cout << "Cache level clipped to [0, 8], originally: " << cache_level << std::endl;
                }
                config.cache_level = std::max(std::min(cache_level, 8), 0);
            } else if (name == "max_node_num") {
                int max_node_num = 0;
                node->QueryIntAttribute("value", &max_node_num);
                if (max_node_num < 1 || max_node_num > 32) {
                    std::cout << "Max node clipped to [1, 32], originally: " << max_node_num << std::endl;
                }
                config.max_node_num = std::max(std::min(max_node_num, 32), 1);
            }
            node = node->NextSiblingElement("integer");
        }

        node = acc_node->FirstChildElement("float");
        while (node) {
            std::string name = node->Attribute("name");
            if (name == "overlap_w") {
                float overlap_w = 0.75f;
                node->QueryFloatAttribute("value", &overlap_w);
                if (overlap_w < 0.5f || overlap_w > 1.5f) {
                    std::cout << "BVH overlap weight should be in range [0.5, 1.5f], clipped." << std::endl;
                    overlap_w = std::clamp(overlap_w, 0.5f, 1.5f);
                }
                config.bvh_overlap_w = overlap_w;
            }
            node = node->NextSiblingElement("float");
        }
    }

    {   // Sensor Element Parsing
        if (const auto node = sensor_node->FirstChildElement("film")) {
            const tinyxml2::XMLElement* film_elem = node->FirstChildElement("integer");
            while (film_elem) {
                std::string name = film_elem->Attribute("name");
                if (name == "width") {
                    film_elem->QueryIntAttribute("value", &config.width);
                } else if (name == "height") {
                    film_elem->QueryIntAttribute("value", &config.height);
                }
                film_elem = film_elem->NextSiblingElement("integer");
            }
            film_elem = node->FirstChildElement("bool");
            while (film_elem) {
                std::string name = film_elem->Attribute("name");
                if (name == "gamma_correction") {
                    film_elem->QueryBoolAttribute("value", &config.gamma_correction);
                }
                film_elem = film_elem->NextSiblingElement("bool");
            }
        }
    }
    return config;
}