/**
 * @file python_render.cpp
 * @author Qianyue He
 * @brief Renderer Nanobind bindings
 * @date 2025-01-10
 * @copyright Copyright (c) 2025
 */
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <memory>
#include "core/xyz.cuh"
#include "renderer/tracer_base.cuh"

class Scene;

namespace nb = nanobind;
class PythonRenderer {
private:
    std::unique_ptr<ColorSpaceXYZ> xyz_host;
    std::unique_ptr<Scene> scene;
    std::unique_ptr<TracerBase> rdr;
public:
    PythonRenderer(const nb::str& xml_path);

    nb::ndarray<nb::numpy, float> render(
        int max_bounce,
        int max_diffuse,
        int max_specular,
        int max_trans,
        bool gamma_corr
    );

    // don't know if this is useful
    void release();     
};