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
 * @author Qianyue He
 * @brief Renderer Nanobind bindings
 * @date 2025.01.10
 */
#pragma once
#include "core/xyz.cuh"
#include "renderer/tracer_base.cuh"
#include <deque>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

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
    std::unique_ptr<SlidingWindowAverage> ftimer; // frame timer
    bool valid;

  public:
    int device_id;

  public:
    PythonRenderer(const nb::str &xml_path, int _device_id, int seed_offset);
    ~PythonRenderer() {
        if (valid) {
            release();
        }
    }

    nb::ndarray<nb::pytorch, float> render();

    nb::ndarray<nb::pytorch, float> variance();

    // don't know if this is useful
    void release();
    void info() const;

    int counter() const { return rdr->cnt(); }
    float avg_frame_time() const { return ftimer->avg(); }
};
