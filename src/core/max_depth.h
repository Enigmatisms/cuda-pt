#pragma once
#include <utility>

struct MaxDepthParams {
    int max_diffuse;
    int max_specular;
    int max_tranmit;
    int max_volume;
    int max_depth;
    float min_time;
    float max_time;

    MaxDepthParams(int max_d = 4, int max_s = 2, int max_t = 8, int max_v = 3, int max_total = 8):
        max_diffuse(max_d),
        max_specular(max_s),
        max_tranmit(max_t),
        max_volume(max_v),
        min_time(0),
        max_time(0),
        max_depth(std::max(std::max(max_total, max_t), std::max(max_d, max_s)))
    {}
};