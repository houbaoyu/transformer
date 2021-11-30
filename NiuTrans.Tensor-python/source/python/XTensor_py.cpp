#include"../tensor/XTensor.h"
#include"../tensor/core/movement/CopyValues.h"
#include"../tensor/core/movement/CopyIndexed.h"
#include"../tensor/core/shape/Squeeze.h"
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace nts;
//get the shape of tensor
py::list XTensor_GetShape(XTensor tensor) {		//三维list的dim：dim[0]=3,dim[1]=2,dim[2]=3；
	py::list shape;
	for (int i = 0; i < tensor.order; i++) {
		shape.append(tensor.dimSize[i]);
	}
	return shape;
}
void TensorRange(XTensor &tensor, DTYPE lower, DTYPE upper, DTYPE step) {
	tensor.Range(lower, upper, step);
}
XTensor TensorSetDataRand(XTensor &tensor, DTYPE lower, DTYPE upper) {
	tensor.SetDataRand(lower, upper);
	return tensor;
}
XTensor TensorSetDataRandn(XTensor &tensor, DTYPE mean, DTYPE standardDeviation) {
	tensor.SetDataRandn(mean, standardDeviation);
	return tensor;
}
//get the item from XTensor by the index in the python
void throwError(int index, int length) {
	std::string errorString = "index ";
	std::ostringstream buff;
	buff << index;
	errorString += buff.str();
	errorString += " out of bounds for dimension 0 ";
	buff.str("");
	buff << (length - 1);
	errorString += "to ";
	errorString += buff.str();
	throw std::runtime_error(errorString);
}
XTensor GetDataFromTensorIndex(XTensor tensor, int index) {

	if (index >= tensor.GetDim(0)) {
		throwError(index, tensor.GetDim(0));
	}
	XTensor retTensor;
	DTYPE retData = tensor.Get1D(index);
	retTensor = NewTensor1D(1, tensor.dataType);
	retTensor.SetDataFixed(retData);
	return retTensor;
}

XTensor GetTensorFromTensorIndex(XTensor tensor, int index) {
	XTensor retTensor;
	if (index >= tensor.GetDim(0)) {
		throwError(index, tensor.GetDim(0));
	}

	XTensor srcIndex = NewTensor1D(1);
	XTensor tgtIndex = NewTensor1D(1);
	int srcIndexData[1] = { index };
	int tgtIndexData[1] = { 0 };
	srcIndex.SetData(srcIndexData, 1);
	tgtIndex.SetData(tgtIndexData, 1);
	retTensor = CopyIndexed(tensor, 0, srcIndex, tgtIndex, 1);
	SqueezeMe(retTensor);
	//DelTensor(srcIndex);
	//DelTensor(tgtIndex);
	//py::print("&&");
	return retTensor;
}
std::string  GetOutputFormat(std::vector<std::string> vecDate, int tOrder, int* tDimSzie, int digit) {
	std::string data = "";
	if (vecDate.empty())
		return data;

	if (tOrder == 1) {			//一维数据，生成 [111,222,ttt]
		data += "[";
		for (int i = 0; i < vecDate.size() - 1; ++i) {	//不包含最后一段数据
			if (vecDate.at(i).size() < digit) {
				data += vecDate.at(i);
				for (int j = 0; j < (digit - vecDate.at(i).size()); ++j) {
					data += "0";				//将剩余的位置补上零
				}
			}
			else
				data += vecDate.at(i);
			data += ", ";					//一段数据的后边用‘，’号分割
		}
		if (vecDate.at(vecDate.size() - 1).size() < digit) {		//如果最后一段数据的长度小于digit
			data += vecDate.at(vecDate.size() - 1);
			for (int j = 0; j < (digit - vecDate.at(vecDate.size() - 1).size()); ++j) {
				data += "0";
			}
		}
		else
			data += vecDate.at(vecDate.size() - 1);
		data += "]";
		return data;
	}
	else if (tOrder == 2) {			//二维数据，生成 [[111,222,ttt],\n\t [111,333,aaa]]
		data += "[[";
		for (int i = 0; i < vecDate.size() - 1; ++i) {
			if (vecDate.at(i).size() < digit) {
				data += vecDate.at(i);
				for (int j = 0; j < (digit - vecDate.at(i).size()); ++j) {
					data += "0";
				}
			}
			else
				data += vecDate.at(i);
			if ((i + 1) % tDimSzie[1] == 0) {			//!!!!!注意此处用的是 tDimSzie[1]
				data += "],\n\t [";
			}
			else {
				data += ", ";
			}
		}
		if (vecDate.at(vecDate.size() - 1).size() < digit) {
			data += vecDate.at(vecDate.size() - 1);
			for (int j = 0; j < (digit - vecDate.at(vecDate.size() - 1).size()); ++j) {
				data += "0";
			}
		}
		else
			data += vecDate.at(vecDate.size() - 1);
		data += "]]";
		return data;
	}
	else if (tOrder == 3) {
		data += "[[[";
		int countDim1 = 0;
		for (int i = 0; i < vecDate.size() - 1; i++) {
			if (vecDate.at(i).size() < digit) {
				data += vecDate.at(i);
				for (int j = 0; j < (digit - vecDate.at(i).size()); ++j) {
					data += "0";
				}
			}
			else
				data += vecDate.at(i);
			if (((i + 1) % tDimSzie[2] == 0) && ((countDim1 + 1) % tDimSzie[1] != 0)) {
				data += "],\n\t  [";
				countDim1++;
			}
			else if (((i + 1) % tDimSzie[2] == 0) && ((countDim1 + 1) % tDimSzie[1] == 0)) {
				data += "]],\n\n\t  [[";
				countDim1++;
			}
			else {
				data += ", ";
			}
		}
		if (vecDate.at(vecDate.size() - 1).size() < digit) {
			data += vecDate.at(vecDate.size() - 1);
			for (int j = 0; j < (digit - vecDate.at(vecDate.size() - 1).size()); ++j) {
				data += "0";
			}
		}
		else
			data += vecDate.at(vecDate.size() - 1);
		data += "]]]";
		return data;
	}
	else {				//三维以上的数据的每一段用‘，’号隔开
		for (int i = 0; i < vecDate.size() - 1; ++i) {
			data += vecDate.at(i);
			data += ", ";
		}
		data += vecDate.at(vecDate.size() - 1);
		return data;
	}
}
//get the data from tensor		获得的string数据格式如下: [[11,22,rr],\n\t[11,11,ww]]
std::string GetTensorData(XTensor*tensor) {
	std::vector<std::string> vecDate;

	int digit = 0;
	int tOrder = tensor->order;
	int* tDimSize = new int[tOrder];
	for (int i = 0; i < tOrder; i++) {
		tDimSize[i] = tensor->GetDim(i);
	}
	if (tensor->dataType == X_DOUBLE)
		for (int i = 0; i < tensor->unitNum; ++i)
		{
			std::ostringstream buff;
			buff << ((double*)tensor->data)[i];		//按照data数组分段写进缓存
			if (buff.str().find('.') != -1) {		//用小数点分割
				vecDate.push_back(buff.str());
				if (digit < int(buff.str().size()))
					digit = int(buff.str().size());
			}
			else
				vecDate.push_back(buff.str() + '.');//此处将整数的最后加上小数点，保证格式统一
		}
	else if (tensor->dataType == X_INT)
		for (int i = 0; i < tensor->unitNum; ++i)
		{
			std::ostringstream buff;
			buff << ((int*)tensor->data)[i];
			vecDate.push_back(buff.str());
		}
	else if (tensor->dataType == X_FLOAT)
		for (int i = 0; i < tensor->unitNum; ++i)
		{
			std::ostringstream buff;
			buff << ((float*)tensor->data)[i];
			if (buff.str().find('.') != -1) {
				if (digit < int(buff.str().size()))
					digit = int(buff.str().size());
				vecDate.push_back(buff.str());
			}
			else
				vecDate.push_back(buff.str() + '.');
		}
	else
		ShowNTErrors("TODO!");

	return GetOutputFormat(vecDate, tOrder, tDimSize, digit);
}
/*some functions for creating a new tensor from list*/
static int to_aten_dim_list(py::list list) {
	//compute the dim of list（计算维度）
	//例如2，3，3三维list:[ [[1,1,1][2,2,2][3,3,3]] [[1,1,1][2,2,2][3,3,3]] ]
	int dim = 1;
	for (auto item_list : list) {
		if (PySequence_Check(item_list.ptr())) {//PySequence_Check：如果参数是序列的返回1，item_list.ptr()智能指针
			return dim + to_aten_dim_list(py::cast<py::list>(item_list));
		}
		else
			return dim;
	}
	return dim;
}

static int* to_aten_shape_list(int dim, py::list list) {
	//compute the shape of list（计算每一个维度的大小），例如2，3，3三维list:[ [[1,1,1][2,2,2][3,3,3]] [[1,1,1][2,2,2][3,3,3]] ]
	int* shape;
	shape = (int*)malloc(sizeof(int) * dim);
	py::list item_list;
	if (dim != 1)
		item_list = py::cast<py::list>(list[0]);
	shape[0] = static_cast<int>(PyObject_Length(list.ptr()));
	for (int i = 1; i < dim - 1; ++i) {
		shape[i] = static_cast<int>(PyObject_Length(item_list.ptr()));//static_cast<int>()强制类型转换，PyObject_Length返回序列中对象的数量
		item_list = py::cast<py::list>(item_list[0]);
	}
	if (dim != 1)
		shape[dim - 1] = static_cast<int>(PyObject_Length(item_list.ptr()));

	return shape;
}

XTensor*create_from_python(py::handle input, py::object aa) {
	PyObject* obj = input.ptr();
	if (PySequence_Check(obj)) {		//判断pyobject是否是连续的序列
		py::list list = py::cast<py::list>(input);	//强制类型转换为list
		py::array_t<float> array = py::cast<py::array>(list);		//array_t只能接受某个数据类型的numpy数组
		auto dim = to_aten_dim_list(list);
		auto shape = to_aten_shape_list(dim, list);
		if (PyNumber_Check(aa.ptr())) {
			py::float_ num = py::cast<py::float_>(aa);
			//py::print(num);
			if (num.operator float() == 1.0) {		//用外部内存
				XTensor*floatTensor = new XTensor(dim, shape, X_FLOAT, -1, (void*)array.data());
				//floatTensor->change_data((void*)array.data());
				array.release();
				//py::print((double*)floatTensor->data);
				return floatTensor;
			}
			else if (num.operator float() == 0.0) {
				XTensor*floatTensor = NewTensor(dim, shape, X_FLOAT, -1);	//产生一个空的tensor
				auto requestArray = array.request();			//获得缓冲区的buffer_info	
				void* data_ptr = requestArray.ptr;				//获得list在缓冲区的数据
				floatTensor->SetData(data_ptr, floatTensor->unitNum);
				//py::print((double*)floatTensor->data);
				//floatTensor.SetData(data_ptr, floatTensor.unitNum);		//将list数据传递给xtensor
				return floatTensor;
			}
			else {
				py::print("erro1");
				return 0;
			}
		}
		else {
			py::print("erro2");
			return 0;
		}
	}
	else {
		py::print("erro3");
		return 0;
	}
}
/*
XTensor*create_from_python(py::array array, py::object aa) {
	PyObject* obj = array.ptr();
	if (PySequence_Check(obj)) {		//判断pyobject是否是连续的序列
		int dim = array.ndim();
		auto shape1 = array.shape();
		int *shape = (int*)malloc(dim);
		for (int i = 0; i < dim; i++) {
			shape[i] = shape1[i];
		}
		if (PyNumber_Check(aa.ptr())) {
			py::float_ num = py::cast<py::float_>(aa);
			if (num.operator float() == 1.0) {		//用外部内存
				XTensor*floatTensor = new XTensor(dim, shape, X_FLOAT, -1, (void*)array.data());
				return floatTensor;
			}
			else if (num.operator float() == 0.0) {
				XTensor*floatTensor = NewTensor(dim, shape, X_FLOAT, -1);	//产生一个空的tensor
				floatTensor->SetData((void*)array.data(), floatTensor->unitNum);
				return floatTensor;
			}
			else {
				py::print("erro1");
				return 0;
			}
		}
		else {
			py::print("erro2");
			return 0;
		}
	}
	else {
		py::print("erro3");
		return 0;
	}
}

XTensor*create_from_python_gpu(py::array array, py::object aa) {
	PyObject* obj = array.ptr();
	if (PySequence_Check(obj)) {		//判断pyobject是否是连续的序列
		int dim = array.ndim();
		auto shape1 = array.shape();
		int *shape = (int*)malloc(dim);
		for (int i = 0; i < dim; i++) {
			shape[i] = shape1[i];
		}
		if (PyNumber_Check(aa.ptr())) {
			py::float_ num = py::cast<py::float_>(aa);
			if (num.operator float() == 1.0) {		//用外部内存
				XTensor*floatTensor = new XTensor(dim, shape, X_FLOAT, 0, (void*)array.data());
				return floatTensor;
			}
			else if (num.operator float() == 0.0) {
				XTensor*floatTensor = NewTensor(dim, shape, X_FLOAT, 0);	//产生一个空的tensor
				floatTensor->SetData((void*)array.data(), floatTensor->unitNum);
				return floatTensor;
			}
			else {
				py::print("erro1");
				return 0;
			}
		}
		else {
			py::print("erro2");
			return 0;
		}
	}
	else {
		py::print("erro3");
		return 0;
	}
}
*/

XTensor*create_from_python_gpu(py::handle input, py::object aa) {
	PyObject* obj = input.ptr();
	if (PySequence_Check(obj)) {		//判断pyobject是否是连续的序列
		py::list list = py::cast<py::list>(input);	//强制类型转换为list
		py::array_t<float> array = py::cast<py::array>(list);		//array_t只能接受某个数据类型的numpy数组
		auto dim = to_aten_dim_list(list);
		auto shape = to_aten_shape_list(dim, list);

		if (PyNumber_Check(aa.ptr())) {
			py::float_ num = py::cast<py::float_>(aa);
			//py::print(num);
			if (num.operator float() == 1.0) {		//用外部内存
				XTensor*floatTensor = new XTensor(dim, shape, X_FLOAT, 0, NULL);
				//floatTensor->change_data((void*)array.data());
				array.release();
				//py::print((double*)floatTensor->data);
				return floatTensor;
			}
			else if (num.operator float() == 0.0) {
				XTensor*floatTensor = NewTensor(dim, shape, X_FLOAT, 0);	//产生一个空的tensor
				auto requestArray = array.request();			//获得缓冲区的buffer_info	
				void* data_ptr = requestArray.ptr;				//获得list在缓冲区的数据
				floatTensor->SetData(data_ptr, floatTensor->unitNum);
				//py::print((double*)floatTensor->data);
				//floatTensor.SetData(data_ptr, floatTensor.unitNum);		//将list数据传递给xtensor
				return floatTensor;
			}
			else {
				py::print("erro1");
				return 0;
			}
		}
		else {
			py::print("erro2");
			return 0;
		}
	}
	else {
		py::print("erro3");
		return 0;
	}
}

void Copy_Tensor(XTensor * a,const XTensor * b) {
	a->Init();
	a->id = MakeTensorID();
	a->ShallowCopy(b);
	a->data = NULL;
	a->dataHost = NULL;

	if (b->isTmp) {
		a->devID = b->devID;
		a->mem = b->mem;
		a->data = b->data;
		a->reserved = b->reserved;
		const_cast<XTensor*>(b)->reserved = 0;
		a->signature = b->signature;
		const_cast<XTensor*>(b)->data = NULL;
	}
	else {
		a->devID = b->devID;
		a->mem = b->mem;
		InitTensorV2(a, b);
		_CopyValues(b, a);
	}
	a->isInit = true;
	a->isTmp = b->isTmp;
}

void init_xtensor(py::module &m) {
	py::class_<XTensor>(m, "XTensor")
		.def_readwrite("id", &XTensor::id)
		.def_readwrite("order", &XTensor::order)
		.def_readwrite("data", &XTensor::data)
		.def_readwrite("dataType", &XTensor::dataType)
		.def_readwrite("enableGrad", &XTensor::enableGrad)
		.def_readwrite("isGrad", &XTensor::isGrad)
		.def_readwrite("isVar", &XTensor::isVar)
		.def_readwrite("grad", &XTensor::grad)
		.def(py::init<>())
		.def(py::init())		//gpu适合此用法
		.def("init", &XTensor::Init)
		.def("shape", &XTensor_GetShape)
		//set data
		.def("Range", &TensorRange)
		.def("SetDataRand", &TensorSetDataRand)
		.def("SetDataRandn", &TensorSetDataRandn)
		.def("GetDim", &XTensor::GetDim, py::return_value_policy::reference)
		.def("__getitem__",
			[](const XTensor& tensor, int index) {
		if (tensor.order == 1) {
			return GetDataFromTensorIndex(tensor, index);
		}
		else {
			return GetTensorFromTensorIndex(tensor, index);
		}
	})
		.def("__repr__",
			[](XTensor &tmp) {
		printf("lalalalahouhouhou:%d\n", tmp.id);
		//XTensor*tmp = NewTensor(tensor.order, tensor.dimSize, X_FLOAT, -1);
		//_CopyValues(&tensor, tmp);
		return "tensor (" + std::to_string(tmp.id) + GetTensorData(&tmp) + ")";
	});
	m.def("create_from_python", &create_from_python, py::return_value_policy::take_ownership);
	m.def("create_from_python_gpu", &create_from_python_gpu, py::return_value_policy::take_ownership);
	m.def("Copy_Tensor", &Copy_Tensor);
}

