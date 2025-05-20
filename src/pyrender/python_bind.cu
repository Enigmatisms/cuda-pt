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

#include "./python_render.cuh"
#include <nanobind/nanobind.h>

const char *CONSTRUCTOR_DOC =
    R"doc(
Initialize the renderer with the given scene path to an XML scene file.

Parameters:
    scene_path (str): Path to the XML scene file.
    device_id (int): CUDA device ID.
    seed_offset (int): For different process, different seed offset should be used
        so that the rendering output is different
)doc";

const char *RENDER_DOC =
    R"doc(
Render a frame using the specified path tracer settings.

Parameters:
    max_bounces (int): Maximum number of path bounces (default: 6).
    max_diffuse (int): Maximum number of diffuse bounces (default: 4).
    max_specular (int): Maximum number of specular bounces (default: 2).
    max_transmit (int): Maximum number of transmission bounces (default: 4).
    gamma_corr (bool): Whether to apply gamma correction to the output (default: False).
Returns:
    torch.Tensor: A tensor containing the rendered image.
)doc";
const char *INFO_DOC = "Print the basic settings of the PythonRenderer.";
const char *RELEASE_DOC = "Release the resource of the PythonRenderer.";
const char *COUNTER_DOC =
    "Return the frames accumulated of the current instance.";
const char *FRAME_TIME_DOC =
    "Return the average frame time of the renderer (in milliseconds).";
const char *VARIANCE_DOC =
    "Return the variance estimation buffer (single channel float32).";

namespace nb = nanobind;

NB_MODULE(pyrender, m) {
    nb::class_<PythonRenderer>(m, "PythonRenderer")
        .def(nb::init<const nb::str &, int, int>(), CONSTRUCTOR_DOC)
        .def("render", &PythonRenderer::render, RENDER_DOC)
        .def("release", &PythonRenderer::release, RELEASE_DOC)
        .def("counter", &PythonRenderer::counter, COUNTER_DOC)
        .def("variance", &PythonRenderer::variance, VARIANCE_DOC)
        .def("avg_frame_time", &PythonRenderer::avg_frame_time, FRAME_TIME_DOC)
        .def("info", &PythonRenderer::info, INFO_DOC);
}
