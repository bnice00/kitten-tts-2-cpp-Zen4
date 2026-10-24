#include "engine.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
// numpy array that owns the vector, no copy
static py::array_t<float> to_numpy(std::vector<float> && audio) {
    auto * owned = new std::vector<float>(std::move(audio));
    py::capsule free_owned(owned, [](void * v) { delete static_cast<std::vector<float> *>(v); });
    return py::array_t<float>(owned->size(), owned->data(), free_owned);
}
PYBIND11_MODULE(_native, m) {
    m.doc() = "KittenTTS2 CPU engine (same code path as the kitten-tts CLI)";
    py::class_<kitten::engine>(m, "Engine")
        .def(py::init<const kitten::args_map &>(), py::arg("args"), py::call_guard<py::gil_scoped_release>())
        .def("metadata", &kitten::engine::metadata)
        .def("normalize", &kitten::engine::normalize, py::arg("text"), py::call_guard<py::gil_scoped_release>())
        .def("generate", [](kitten::engine & e, const kitten::args_map & args, py::object callback) {
            kitten::chunk_callback on_chunk;
            if (!callback.is_none()) {
                on_chunk = [&callback](const std::vector<float> & audio) {
                    py::gil_scoped_acquire gil;
                    return py::bool_(callback(to_numpy(std::vector<float>(audio)))).cast<bool>();
                };
            }
            std::string report;
            std::vector<float> audio;
            {
                py::gil_scoped_release nogil;
                audio = e.generate(args, report, on_chunk);
            }
            return py::make_tuple(to_numpy(std::move(audio)), report);
        }, py::arg("args"), py::arg("callback") = py::none());
}
