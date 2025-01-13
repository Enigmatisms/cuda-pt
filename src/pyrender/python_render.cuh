/**
 * @file python_render.cpp
 * @author Qianyue He
 * @brief Renderer Nanobind bindings
 * @date 2025-01-10
 * @copyright Copyright (c) 2025
 */
#pragma once
#include <deque>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "core/xyz.cuh"
#include "renderer/tracer_base.cuh"

class Scene;

class SlidingWindowAverage {
public:
    explicit SlidingWindowAverage(size_t max_num = 12)
        : max_num_(max_num), sum_(0.0f) {}

    void record(float value) {
        if (window_.size() == static_cast<size_t>(max_num_)) {
            sum_ -= window_.front();
            window_.pop_front();
        }

        window_.push_back(value);
        sum_ += value;
    }

    float avg() const {
        return window_.empty() ? 0.0f : sum_ / float(window_.size());
    }
private:
    float sum_;
    size_t max_num_;
    std::deque<float> window_; 
};

namespace nb = nanobind;
class PythonRenderer {
private:
    std::unique_ptr<ColorSpaceXYZ> xyz_host;
    std::unique_ptr<Scene> scene;
    std::unique_ptr<TracerBase> rdr;
    std::unique_ptr<SlidingWindowAverage> ftimer;   // frame timer
    bool valid;
public:
    int device_id;
public:
    PythonRenderer(const nb::str& xml_path, int _device_id, int seed_offset);
    ~PythonRenderer() {
        if (valid) {
            release();
        }
    }

    nb::ndarray<nb::pytorch, float> render(
        int max_bounce,
        int max_diffuse,
        int max_specular,
        int max_trans,
        bool gamma_corr
    );

    nb::ndarray<nb::pytorch, float> variance();

    // don't know if this is useful
    void release(); 
    void info() const;

    int counter() const {
        return rdr->cnt();
    }
    float avg_frame_time() const {
        return ftimer->avg();
    }
};