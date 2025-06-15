#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "nbody.h"

PYBIND11_MODULE(nbody_cpp, m)
{
     m.doc() = "Direct N-body gravitational simulation in C++";

     pybind11::class_<NBodySimulation>(m, "NBodySimulation")
         .def(pybind11::init<const std::vector<double> &,
                             const std::vector<std::vector<double>> &,
                             const std::vector<std::vector<double>> &,
                             double>(),
              pybind11::arg("masses"),
              pybind11::arg("positions"),
              pybind11::arg("velocities"),
              pybind11::arg("G") = 1.0)
         .def("simulate", &NBodySimulation::simulate,
              pybind11::arg("t_end"),
              pybind11::arg("dt"))
         .def("total_energy", &NBodySimulation::total_energy)
         .def("get_history", &NBodySimulation::get_history)
         .def("get_positions", &NBodySimulation::get_positions)
         .def("get_velocities", &NBodySimulation::get_velocities)
         .def("get_masses", &NBodySimulation::get_masses)
         .def("get_n_bodies", &NBodySimulation::get_n_bodies)
         .def("get_G", &NBodySimulation::get_G)
         .def("get_time_steps", &NBodySimulation::get_time_steps);
}