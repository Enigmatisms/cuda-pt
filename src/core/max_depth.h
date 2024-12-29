#pragma once
#include <utility>

struct MaxDepthParams {
    int max_diffuse;
    int max_specular;
    int max_tranmit;
    int max_depth;

    MaxDepthParams(int max_d = 4, int max_s = 2, int max_t = 8, int max_total = 8):
        max_diffuse(max_d),
        max_specular(max_s),
        max_tranmit(max_t),
        max_depth(std::max(std::max(max_total, max_t), std::max(max_d, max_s)))
    {}
};