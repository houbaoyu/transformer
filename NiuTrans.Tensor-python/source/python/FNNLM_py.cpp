/*-----------------------------------------pybind11封装内容-------------------------------------*/
#include"../sample/fnnlm/FNNLM.h"
#include"../tensor/XGlobal.h"
#include "../tensor/XUtility.h"
#include "../tensor/XDevice.h"
#include "../tensor/function/FHeader.h"
#include "../network/XNet.h"
#include "../tensor/core/CHeader.h"
#include <pybind11/pybind11.h>
#include <iostream>
namespace py = pybind11;
using namespace fnnlm;
using namespace nts;
#define MAX_NAME_LENGTH 1024
#define MAX_LINE_LENGTH_HERE 1024 * 32


char trainFN[MAX_NAME_LENGTH] = "";   // file name of the training data
char modelFN[MAX_NAME_LENGTH] = "";   // file name of the FNN model
char testFN[MAX_NAME_LENGTH] = "";    // file name of the test data
char outputFN[MAX_NAME_LENGTH] = "";  // file name of the result data
float learningRate = 0.01F;           // learning rate
int nStep = 10000000;                   // max learning steps (or model updates)
int nEpoch = 10;                      // max training epochs
float minmax = 0.08F;                 // range [-p,p] for parameter initialization
int sentBatch = 0;                    // batch size at the sentence level
int wordBatch = 1;                    // batch size at the word level
bool shuffled = false;                // shuffled the training data file or not
bool autoDiff = true;                // indicator of automatic differentiation
int line_sum = 0;

void PyLoadArgs(py::int_ argc, py::list argvs, FNNModel & model) {
	char*argv[200];
	int n = 0;
	for (auto item : argvs) {
		std::string s = py::cast<py::str>(item);
		argv[n] = new char[s.length() + 1];
		strcpy(argv[n], s.c_str());
		argv[n][s.length()] = '\0';
		n += 1;
		
	}
	LoadArgs(argc, argv, model);
}
//get the shape of tensor
py::list GetShape(XTensor tensor) {		//三维list的dim：dim[0]=3,dim[1]=2,dim[2]=3；
	py::list shape;
	for (int i = 0; i < tensor.order; i++) {
		shape.append(tensor.dimSize[i]);
	}
	return shape;
}
//get the shape of hiddenW or hiddenB
py::list GetModelShape_hiddenW(FNNModel model) {
	py::list shape;
	shape = GetShape(model.hiddenW);
	return shape;
}
py::list GetModelShape_hiddenB(FNNModel model) {
	py::list shape;
	shape = GetShape(model.hiddenB);
	return shape;
}
void PyClear(FNNModel &model, py::bool_ isNodeGrad) {
	bool is = isNodeGrad;
	Clear(model, is);
}
void PyTrain(py::str train, py::bool_ isShuffled, FNNModel &model) {
	std::string s = train;
	char *c = new char[s.length() + 1];
	strcpy(c, s.c_str());
	c[s.length()] = '\0';
	Train(c, isShuffled, model);
}
void PyDump(py::str py_fn, FNNModel &model) {
	std::string s = py_fn;
	char *fn = new char[s.length() + 1];
	strcpy(fn, s.c_str());
	fn[s.length()] = '\0';
	Dump(fn, model);
}
void PyRead(py::str py_fn, FNNModel &model) {
	std::string s = py_fn;
	char *fn = new char[s.length() + 1];
	strcpy(fn, s.c_str());
	fn[s.length()] = '\0';
	Read(fn, model);
}
void PyTest(py::str py_test, py::str py_result, FNNModel &model) {
	std::string s1 = py_test;
	char *test = new char[s1.length() + 1];
	strcpy(test, s1.c_str());
	test[s1.length()] = '\0';
	std::string s2 = py_result;
	char *result = new char[s2.length() + 1];
	strcpy(result, s2.c_str());
	result[s2.length()] = '\0';
	Test(test, result, model);
}
void PyShuffle(py::str py_srcFile, py::str py_tgtFile) {
	std::string s1 = py_srcFile;
	char *srcFile = new char[s1.length() + 1];
	strcpy(srcFile, s1.c_str());
	srcFile[s1.length()] = '\0';
	std::string s2 = py_tgtFile;
	char *tgtFile = new char[s2.length() + 1];
	strcpy(tgtFile, s2.c_str());
	tgtFile[s2.length()] = '\0';
	Shuffle(srcFile, tgtFile);
}
int dingwei(FILE * file) {
	char lineBuf1[MAX_LINE_LENGTH_HERE];
	for (int i = 0; i < line_sum; i++) {
		fgets(lineBuf1, MAX_LINE_LENGTH_HERE - 1, file);
	}
	return 1;
}
//返回三元组（input，gold，ngramNum）
py::tuple PyBatch(FNNModel &model) {
	NGram * ngrams = new NGram[MAX_LINE_LENGTH_HERE];
	FILE * file = fopen(trainFN, "rb");
	CheckErrors(file, "Cannot open the training file");
	dingwei(file);
	int ngramNum = 1;
	ngramNum = LoadNGrams(file, model.n, ngrams, sentBatch, wordBatch);
	py::list input_py;
	XTensor gold;
	if (ngramNum > 0) {
		XTensor inputs[MAX_N_GRAM];
		/* make the input tensor for position i */
		for (int i = 0; i < model.n - 1; i++) {
			MakeWordBatch(inputs[i], ngrams, ngramNum, i, model.vSize, model.devID);
			input_py.append(inputs[i]);
		}
		MakeWordBatch(gold, ngrams, ngramNum, model.n - 1, model.vSize, model.devID);
	}
	fclose(file);
	return py::make_tuple(input_py, gold, ngramNum);
}
XTensor PyForward(py::list input_py, FNNModel &model, FNNNet net) {
	XTensor output;
	XTensor inputs[MAX_N_GRAM];
	int n = 0;
	for (auto item : input_py) {
		inputs[n] = XTensor(py::cast<XTensor>(item));
		n += 1;
	}
	Forward(inputs, output, model, net);
	return output;
}
void PyBackward(py::list input_py, XTensor &output, XTensor &gold, FNNModel &model, FNNModel &grad, FNNNet &net) {
	XTensor inputs[MAX_N_GRAM];
	int n = 0;
	for (auto item : input_py) {
		inputs[n] = XTensor(py::cast<XTensor>(item));
		n += 1;
	}
	Backward(inputs, output, gold, CROSSENTROPY, model, grad, net);
}
void PyUpdate(FNNModel &model, FNNModel &grad, py::float_ epsilon, py::bool_ isNodeGrad) {
	float ee = epsilon;
	bool rr = isNodeGrad;
	Update(model, grad, ee, rr);
}

void PyForwardAutoDiff(py::list ngram, int batch , XTensor &output, FNNModel &model) {
	
	int n = model.n;
	int depth = model.hDepth;
	int size = batch*(n - 1);
	int * index = new int[size];
	int ii = 0;
	for (auto item : ngram) {
		int jj = 0;
		for (auto i : item) {
			if (jj < model.n-1) {
				int a = ii * (n - 1) + jj;
				py::int_ py_num = py::cast<py::int_>(i);
				index[a] = py_num;
				jj++;
			}
		}
		ii++;
	}
	XTensor words;
	XTensor embeddingBig;
	XTensor hidden;
	XTensor b;
	InitTensor1D(&words, size, X_INT, model.devID);
	words.SetData(index, size);
	embeddingBig = Gather(model.embeddingW, words);


	delete[] index;

	int dimSize[2];
	dimSize[0] = embeddingBig.GetDim(0) / (n - 1);
	dimSize[1] = embeddingBig.GetDim(1) * (n - 1);

	hidden = Reshape(embeddingBig, embeddingBig.order, dimSize);

	for (int i = 0; i < depth; i++)
		hidden = HardTanH(MMul(hidden, model.hiddenW[i]) + model.hiddenB[i]);


	output = Softmax(MMul(hidden, model.outputW) + model.outputB, 1);
}
XTensor PyCrossEntropy(XTensor& output, XTensor& gold) {
	return CrossEntropy(output, gold);
}
py::float_ PyGetProb(XTensor& output, XTensor& gold) {
	XTensor * s = NULL;
	py::float_ prob = GetProb(output, gold,s);
	return prob;
}
py::float_ Py_ReduceSumAll(XTensor &lossTensor) {
	float prob;
	_ReduceSumAll(&lossTensor, &prob);
	py::float_ prob_py = prob;
	return prob_py;
}
XTensor PyEmbedding(py::list ngram, FNNModel &model) {
	int n = model.n;
	int depth = model.hDepth;
	int size = len(ngram)*(n - 1);
	int * index = new int[size];
	int ii = 0;
	for (auto item : ngram) {
		int jj = 0;
		for (auto i : item) {
			if (jj < model.n - 1) {
				int a = ii * (n - 1) + jj;
				py::int_ py_num = py::cast<py::int_>(i);
				index[a] = py_num;
				jj++;
			}
		}
		ii++;
	}
	
	XTensor words;
	XTensor embeddingBig;
	XTensor hidden;
	XTensor b;
	InitTensor1D(&words, size, X_INT, model.devID);
	words.SetData(index, size);
	embeddingBig = Gather(model.embeddingW, words);



	delete[] index;

	int dimSize[2];
	dimSize[0] = embeddingBig.GetDim(0) / (n - 1);
	dimSize[1] = embeddingBig.GetDim(1) * (n - 1);

	hidden = Reshape(embeddingBig, embeddingBig.order, dimSize);
	return hidden;
}
XTensor PyLinear(const XTensor &xtensor,FNNModel & model,int i) {
	XTensor tmp = MMul(xtensor, model.hiddenW[i],1.0,NULL);
	XTensor tmp1 = tmp + model.hiddenB[i];
	return tmp1;
}
XTensor PyLinear2(const XTensor &xtensor, FNNModel & model) {
	XTensor tmp = MMul(xtensor, model.outputW, 1.0, NULL);
	XTensor tmp1 = tmp + model.outputB;
	return tmp1;
}
py::list List_hiddenW(FNNModel &model) {
	py::list tmp;
	for (int i = 0; i < model.hDepth; i++) {
		tmp.append(model.hiddenW[i]);
	}
	return tmp;
}
py::list List_hiddenB(FNNModel &model) {
	py::list tmp;
	for (int i = 0; i < model.hDepth; i++) {
		tmp.append(model.hiddenB[i]);
	}
	return tmp;
}
XTensor PySoftmax(const XTensor &x, int leadDim) {
	return Softmax(x, leadDim);
}
XTensor PyHardTanH(const XTensor &x) {
	return HardTanH(x);
}
void ENABLE_GRAD_fun() {
	X_ENABLE_GRAD = true;
}
bool check_xtensor(XTensor & x1, XTensor & x2) {
	std::cout << "fun_in111" << std::endl;
	if (x1.dataType != x2.dataType) {
		std::cout << "type-wrong" << std::endl;
		return false;
	}
	if (x1.dataType == X_DOUBLE)
		for (int i = 0; i < x1.unitNum; ++i)
		{
			double d1 = ((double*)x1.data)[i];
			double d2 = ((double*)x2.data)[i];
			if (d1 != d2)
				return false;
		}
	else if (x1.dataType == X_INT)
		for (int i = 0; i < x1.unitNum; ++i)
		{
			int d1 = ((int*)x1.data)[i];
			int d2 = ((int*)x2.data)[i];
			if (d1 != d2)
				return false;
		}
	else if (x1.dataType == X_FLOAT) {
		std::cout << "fun_in222" << std::endl;
		for (int i = 0; i < x1.unitNum; ++i)
		{
			std::cout << x1.data << std::endl;
			float d1 = ((float*)x1.data)[i];
			std::cout << d1 << std::endl;
			float d2 = ((float*)x2.data)[i];
			std::cout << d1 <<","<<d2<< std::endl;
			if (d1 != d2)
				return false;
		}
		
	}
		
	return true;
}
void PyFNNLMMain(int argc,py::list arg_py) {
	char*argv[200];
	int n = 0;
	for (auto item : arg_py) {
		std::string s = py::cast<py::str>(item);
		argv[n] = new char[s.length() + 1];
		strcpy(argv[n], s.c_str());
		argv[n][s.length()] = '\0';
		n += 1;

	}
	FNNLMMain(argc, argv);
}
void X_ENABLE_GRAD_true() {
	X_ENABLE_GRAD = true;
}
void init_fnnlm(py::module & m) {
	py::class_<FNNModel>(m, "FNNModel")
		.def_readwrite("embeddingW", &FNNModel::embeddingW)
		.def_readwrite("outputW", &FNNModel::outputW)
		.def_readwrite("outputB", &FNNModel::outputB)
		.def_readwrite("n", &FNNModel::n)
		.def_readwrite("eSize", &FNNModel::eSize)
		.def_readwrite("hDepth", &FNNModel::hDepth)
		.def_readwrite("hSize", &FNNModel::hSize)
		.def_readwrite("vSize", &FNNModel::vSize)
		.def(py::init<>())
		.def("hiddenWshape", &GetModelShape_hiddenW)
		.def("hiddenBshape", &GetModelShape_hiddenB);
	py::class_<FNNNet>(m, "FNNNet")
		.def(py::init<>());
	m.def("LoadArgs", &PyLoadArgs);
	m.def("Init", &Init);
	m.def("Check", &Check);
	m.def("Copy", &Copy);
	m.def("Clear", &PyClear);
	m.def("Train", &PyTrain);
	m.def("Dump", &PyDump);
	m.def("Read", &PyRead);
	m.def("Test", &PyTest);
	m.def("FNNLMMain", &PyFNNLMMain);

	//train函数展开
	m.def("Shuffle", &PyShuffle);
	m.def("Batch", &PyBatch, py::return_value_policy::take_ownership);
	m.def("Forward", &PyForward, py::return_value_policy::take_ownership);
	m.def("Backward", &PyBackward);
	m.def("Update", &PyUpdate);
	m.def("ForwardAutoDiff", &PyForwardAutoDiff, py::return_value_policy::reference);
	m.def("CrossEntropy", &PyCrossEntropy, py::return_value_policy::take_ownership);
	m.def("GetProb", &PyGetProb, py::return_value_policy::reference);
	m.def("ReduceSumAll", &Py_ReduceSumAll, py::return_value_policy::reference);

	//forward函数展开
	m.def("Embedding",&PyEmbedding, py::return_value_policy::take_ownership);
	m.def("MMul", py::overload_cast<const XTensor &, const XTensor &, DTYPE, XPRunner *>(&MatrixMul), py::return_value_policy::reference);
	m.def("HardTanH",&PyHardTanH, py::return_value_policy::reference);
	m.def("Softmax", &PySoftmax, py::return_value_policy::reference);
	m.def("Linear",&PyLinear, py::return_value_policy::reference);
	m.def("Linear2", &PyLinear2, py::return_value_policy::reference);
	m.def("List_hiddenW", &List_hiddenW, py::return_value_policy::reference);
	m.def("List_hiddenB", &List_hiddenB, py::return_value_policy::reference);
	m.def("ENABLE_GRAD_fun", &ENABLE_GRAD_fun);
	m.def("check_xtensor", &check_xtensor, py::return_value_policy::reference);

}
