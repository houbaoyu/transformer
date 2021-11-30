/*---------------------------------------------pybind11封装内容------------------------------------------*/
#include"../network/XNet.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace nts;
void init_xnet(py::module &m) {
	py::class_<XNet>(m, "XNet")
		.def_readwrite("id", &XNet::id)
		.def_readwrite("nodes", &XNet::nodes)
		.def_readwrite("gradNodes", &XNet::gradNodes)
		.def_readwrite("outputs", &XNet::outputs)
		.def_readwrite("inputs", &XNet::inputs)
		.def_readwrite("isGradEfficient", &XNet::isGradEfficient)
		.def("Backward", (void (XNet::*)(XTensor&)) &XNet::Backward)
		.def("Clear", &XNet::Clear)
		.def(py::init<>());
}