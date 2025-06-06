// Copyright (C) 2025 Qianyue He
//
// This program is free software: you can redistribute it and/or
// modify it under the terms of the GNU Affero General Public License
// as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
// the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General
// Public License along with this program. If not, see
//
//             <https://www.gnu.org/licenses/>.

/**
 * @author: Qianyue He
 * @brief Rendering configuration
 * @date:   2024.5.24
 */
#include "core/config.h"
#include <algorithm>
#include <iostream>

RenderingConfig
RenderingConfig::from_xml(const tinyxml2::XMLElement *acc_node,
                          const tinyxml2::XMLElement *render_node,
                          const tinyxml2::XMLElement *sensor_node) {
    RenderingConfig config;
    { // Renderer Element Parsing
        const tinyxml2::XMLElement *node =
            render_node->FirstChildElement("integer");
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

    { // Accelerator Element Parsing
        const tinyxml2::XMLElement *node =
            acc_node->FirstChildElement("integer");
        while (node) {
            std::string name = node->Attribute("name");
            if (name == "cache_level") {
                int cache_level = 0;
                node->QueryIntAttribute("value", &cache_level);
                if (cache_level < 0 || cache_level > 8) {
                    std::cout << "Cache level clipped to [0, 8], originally: "
                              << cache_level << std::endl;
                }
                config.bvh.cache_level = std::max(std::min(cache_level, 8), 0);
            } else if (name == "max_node_num") {
                int max_node_num = 0;
                node->QueryIntAttribute("value", &max_node_num);
                if (max_node_num < 1 || max_node_num > 255) {
                    std::cout << "Max node clipped to [1, 255], originally: "
                              << max_node_num << std::endl;
                }
                config.bvh.max_node_num = std::clamp(max_node_num, 1, 255);
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
                    std::cout << "BVH overlap weight should be in range [0.5, "
                                 "1.5f], clipped."
                              << std::endl;
                    overlap_w = std::clamp(overlap_w, 0.5f, 1.5f);
                }
                config.bvh.bvh_overlap_w = overlap_w;
            }
            node = node->NextSiblingElement("float");
        }

        node = acc_node->FirstChildElement("bool");
        while (node) {
            std::string name = node->Attribute("name");
            if (name == "use_sbvh") {
                node->QueryBoolAttribute("value", &config.bvh.use_sbvh);
            } else if (name == "use_ref_unsplit") {
                node->QueryBoolAttribute("value", &config.bvh.use_ref_unsplit);
            }
            node = node->NextSiblingElement("bool");
        }
    }

    { // Sensor Element Parsing
        if (const auto node = sensor_node->FirstChildElement("film")) {
            const tinyxml2::XMLElement *film_elem =
                node->FirstChildElement("integer");
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
                    film_elem->QueryBoolAttribute("value",
                                                  &config.gamma_correction);
                }
                film_elem = film_elem->NextSiblingElement("bool");
            }
        }
    }
    return config;
}
