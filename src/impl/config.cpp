/**
 * Rendering configuration
 * @author: Qianyue He
 * @date:   2024.5.24
*/

#include "core/config.h"

RenderingConfig RenderingConfig::from_xml(const tinyxml2::XMLElement *sensor_node) {
    RenderingConfig config;
    const tinyxml2::XMLElement *node = sensor_node->FirstChildElement("integer");
    while (node) {
        std::string name = node->Attribute("name");
        if (name == "sample_count")
            node->QueryIntAttribute("value", &config.spp);
        else if (name == "max_bounce")
            node->QueryIntAttribute("value", &config.max_depth);
        node = node->NextSiblingElement("integer");
    }

    node = sensor_node->FirstChildElement("film");
    if (node) {
        const tinyxml2::XMLElement* film_elem = node->FirstChildElement("integer");
        while (film_elem) {
            std::string name = film_elem->Attribute("name");
            if (name == "width") {
                film_elem->QueryIntAttribute("value", &config.width);
            } else if (name == "height") {
                film_elem->QueryIntAttribute("value", &config.height);
            } else if (name == "specular_constraint") {
                film_elem->QueryIntAttribute("value", &config.spec_constraint);
            }
            film_elem = film_elem->NextSiblingElement("integer");
        }
        film_elem = node->FirstChildElement("bool");
        while (film_elem) {
            std::string name = film_elem->Attribute("name");
            if (name == "gamma_correction") {
                film_elem->QueryBoolAttribute("value", &config.gamma_correction);
            } else if (name == "bidirectional") {
                film_elem->QueryBoolAttribute("value", &config.bidirectional);
            }
            film_elem = film_elem->NextSiblingElement("bool");
        }
        film_elem = node->FirstChildElement("float");
        while (film_elem) {
            std::string name = film_elem->Attribute("name");
            if (name == "caustic_scaling") {
                film_elem->QueryFloatAttribute("value", &config.caustic_scaling);
            }
            film_elem = film_elem->NextSiblingElement("float");
        }
    }
    return config;
}