#include <nanobind/nanobind.h>
#include "./python_render.cuh"

const char* CONSTRUCTOR_DOC = "Initialize the renderer with the given scene path to an XML scene file.";
const char* RENDER_DOC      = \
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
const char* RELEASE_DOC    = "Release the resource of the PythonRenderer.";

namespace nb = nanobind;

NB_MODULE(pyrender, m) {
    nb::class_<PythonRenderer>(m, "PythonRenderer")
        .def(nb::init<const nb::str &>(), CONSTRUCTOR_DOC)
        .def("render", &PythonRenderer::render,
            nb::arg("max_bounces") = 6,
            nb::arg("max_diffuse") = 4,
            nb::arg("max_specular") = 2,
            nb::arg("max_transmit") = 4,
            nb::arg("gamma_corr") = false, RENDER_DOC
        )
        .def("release", &PythonRenderer::release, RELEASE_DOC);
}