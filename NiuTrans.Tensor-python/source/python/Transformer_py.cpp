/*-----------------------------------------pybind11封装内容-------------------------------------*/
#include <pybind11/pybind11.h>
#include <iostream>
#include "../sample/transformer/NMT.h"
#include"../sample/transformer/Model.h"
#include"../sample/transformer/Encoder.h"
#include"../sample/transformer/Decoder.h"
#include"../sample/transformer/train/Trainer.h"
#include"../sample/transformer/translate/Translator.h"
#include"../sample/transformer/submodel/CommonModules.h"
#include"../tensor/XUtility.h"
#include"../sample/transformer/Utility.h"
namespace py = pybind11;
using namespace nmt;
enum { NONE, SELF_ATT, EN_DE_ATT };
class Transformer_py{
public:
	py::dict Config_dict(const Config & config);
	void Cache_disable(Model & model);
	XTensor PyEmbedding_En(Model & model, XTensor & input, py::bool_ isDec, py::bool_ isTraining, int nstep);
	XTensor PyEmbedding_De(Model & model, XTensor & input, py::bool_ isDec, py::bool_ isTraining, int nstep);
	py::int_ GetEncoderLayers(Model * model);
	py::int_ GetDecoderLayers(Model * model);
	XTensor EncLayerNorm_att(XTensor& input, Model & model, py::int_ i, bool before, bool after);
	XTensor DecLayerNorm_att(XTensor& input, Model & model, py::int_ i, bool before, bool after);
	XTensor EnDecLayerNorm_att(XTensor& input, Model & model, py::int_ i, bool before, bool after);
	XTensor EncLayerNorm_fnn(XTensor& input, Model & model, py::int_ i, bool before, bool after);
	XTensor DecLayerNorm_fnn(XTensor& input, Model & model, py::int_ i, bool before, bool after);
	XTensor LayerNorm_encoder(XTensor& input, Model & model);
	XTensor LayerNorm_decoder(XTensor& input, Model & model);
	XTensor GetMask(Model &model, XTensor & input);
	XTensor EncoderAttn(Model & model,XTensor& k, XTensor& q, XTensor& v, XTensor* mask, py::bool_ isTraining, py::int_ i);
	XTensor DecoderAttn(Model & model, XTensor& k, XTensor& q, XTensor& v, XTensor* mask, py::bool_ isTraining, py::int_ i);
	XTensor EnDecoderAttn(Model & model, XTensor& k, XTensor& q, XTensor& v, XTensor* mask, py::bool_ isTraining, py::int_ i);
	XTensor Dropout_trans(const XTensor &x, py::float_ dropoutp);
	py::float_ GetEncoderDropoutP(Model * model);
	py::float_ GetDecoderDropoutP(Model * model);
	XTensor FNNMake_En(Model & model,py::int_ i, XTensor& input, bool isTraining);
	XTensor FNNMake_De(Model & model, py::int_ i, XTensor& input, bool isTraining);
	void MakeCacheDisable(Model & model);
	void MakeOutputLayer(Model & model, XTensor& input, XTensor& output, bool isTraining, bool normalized);
	XTensor IndexToOnehot_py(Trainer & trainer, const XTensor & index);
	int LoadConfigFromFile(Config & config, py::str configFN_s, py::list args);
	void Translate_Fun(Config & config);
	Config ConfigInit(int argc, py::list argv);
	void Init_batchLoader(Trainer & trainer, py::str trainfn, py::bool_ training);
	py::tuple LoadBatch(Trainer & trainer, XTensor* batchEnc, XTensor* paddingEnc,
		XTensor* batchDec, XTensor* paddingDec, XTensor* label, py::int_ minSentBatch, py::int_ batchSize, py::int_ devID);
	void MakeCheckpoint(Trainer & trainer, Config & config, Model* model, py::str ss, int id);
	void NMTMain_py(py::int_ argc_py, py::list argv_py);
	void ModelDump(Model * model, py::str modelFN);
};
void Transformer_py::ModelDump(Model * model, py::str modelFN) {
	char cc[1024];
	std::string s = modelFN;
	strcpy(cc, s.c_str());
	cc[s.length()] = '\0';
	model->Dump(cc);
}
void Transformer_py::NMTMain_py(py::int_ argc_py, py::list argv_py) {
	char*arg[300];
	int n = 0;
	for (auto item : argv_py) {
		std::string s = py::cast<py::str>(item);
		arg[n] = new char[s.length() + 1];
		strcpy(arg[n], s.c_str());
		arg[n][s.length()] = '\0';
		n += 1;
	}
	int argc_c = argc_py;
	NMTMain(argc_c,arg);
}
py::dict Transformer_py::Config_dict(const Config & config) {
	py::dict dic;
	dic["modelFN"] = py::str(config.modelFN);
	dic["srcVocabFN"] = py::str(config.srcVocabFN);
	dic["tgtVocabFN"] = py::str(config.tgtVocabFN);
	dic["testFN"] = py::str(config.testFN);
	dic["outputFN"] = py::str(config.outputFN);
	dic["trainFN"] = py::str(config.trainFN);
	dic["validFN"] = py::str(config.validFN);
	dic["devID"] = py::int_(config.devID);
	dic["beamSize"] = py::int_(config.beamSize);
	dic["wBatchSize"] = py::int_(config.wBatchSize);
	dic["sBatchSize"] = py::int_(config.sBatchSize);
	dic["nhead"] = py::int_(config.nhead);
	dic["nEncLayer"] = py::int_(config.nEncLayer);
	dic["nDecLayer"] = py::int_(config.nDecLayer);
	dic["maxRP"] = py::int_(config.maxRP);
	dic["embSize"] = py::int_(config.embSize);
	dic["modelSize"] = py::int_(config.modelSize);
	dic["maxPosLen"] = py::int_(config.maxPosLen);
	dic["fnnHiddenSize"] = py::int_(config.fnnHiddenSize);
	dic["srcVocabSize"] = py::int_(config.srcVocabSize);
	dic["tgtVocabSize"] = py::int_(config.tgtVocabSize);
	dic["padID"] = py::int_(config.padID);
	dic["startID"] = py::int_(config.startID);
	dic["endID"] = py::int_(config.endID);
	dic["preNorm"] = py::bool_(config.preNorm);
	dic["isMT"] = py::bool_(config.isMT);
	dic["shareAllEmbeddings"] = py::int_(config.shareAllEmbeddings);
	dic["shareDecInputOutputWeight"] = py::int_(config.shareDecInputOutputWeight);
	dic["useFP16"] = py::bool_(config.useFP16);
	dic["useRPR"] = py::bool_(config.useRPR);
	dic["isTraining"] = py::bool_(config.isTraining);
	dic["dropout"] = py::float_(config.dropout);
	dic["fnnDropout"] = py::float_(config.fnnDropout);
	dic["attDropout"] = py::float_(config.attDropout);
	dic["lenAlpha"] = py::float_(config.lenAlpha);
	dic["maxLenAlpha"] = py::float_(config.maxLenAlpha);
	dic["lrate"] = py::float_(config.lrate);
	dic["lrbias"] = py::float_(config.lrbias);
	dic["nepoch"] = py::int_(config.nepoch);
	dic["nstep"] = py::int_(config.nstep);
	dic["maxCheckpoint"] = py::int_(config.maxCheckpoint);
	dic["useAdam"] = py::bool_(config.useAdam);
	dic["adamBeta1"] = py::float_(config.adamBeta1);
	dic["adamBeta2"] = py::float_(config.adamBeta2);
	dic["adamDelta"] = py::float_(config.adamDelta);
	dic["nwarmup"] = py::int_(config.nwarmup);
	dic["isShuffled"] = py::bool_(config.isShuffled);
	dic["labelSmoothingP"] = py::float_(config.labelSmoothingP);
	dic["nStepCheckpoint"] = py::int_(config.nStepCheckpoint);
	dic["useEpochCheckpoint"] = py::bool_(config.useEpochCheckpoint);
	dic["updateStep"] = py::int_(config.updateStep);
	dic["isLenSorted"] = py::bool_(config.isLenSorted);
	dic["bufSize"] = py::int_(config.bufSize);
	dic["isDoubledEnd"] = py::bool_(config.isDoubledEnd);
	dic["isSmallBatch"] = py::bool_(config.isSmallBatch);
	dic["isBigBatch"] = py::bool_(config.isBigBatch);
	dic["isRandomBatch"] = py::bool_(config.isRandomBatch);
	dic["bucketSize"] = py::int_(config.bucketSize);
	return dic;
}
void Transformer_py::Cache_disable(Model & model) {
	for (int i = 0; i < model.decoder->nlayer; i++) {
		model.decoder->selfAttCache[i].enable = false;
		model.decoder->enDeAttCache[i].enable = false;
	}
}
XTensor Transformer_py::PyEmbedding_En(Model & model, XTensor & input, py::bool_ isDec, py::bool_ isTraining, int nstep) {
	XTensor x;
	x = model.encoder->embedder.Make(input, isDec, isTraining,nstep);
	return x;
}
XTensor Transformer_py::PyEmbedding_De(Model & model, XTensor & input, py::bool_ isDec, py::bool_ isTraining, int nstep) {
	XTensor x;
	x = model.decoder->embedder.Make(input, isDec, isTraining, nstep);
	return x;
}
py::int_ Transformer_py::GetEncoderLayers(Model * model) {
	py::int_ layers;
	layers = model->encoder->nlayer;
	return layers;
}
py::int_ Transformer_py::GetDecoderLayers(Model * model) {
	py::int_ layers;
	layers = model->decoder->nlayer;
	return layers;
}
XTensor Transformer_py::EncLayerNorm_att(XTensor& input, Model & model, py::int_ i, bool before, bool after) {
	return LayerNorm(input, model.encoder->attLayerNorms[i], model.encoder->preNorm, before, after);
}
XTensor Transformer_py::DecLayerNorm_att(XTensor& input, Model & model, py::int_ i, bool before, bool after) {
	return LayerNorm(input, model.decoder->selfAttLayerNorms[i], model.decoder->preNorm, before, after);
}
XTensor Transformer_py::EnDecLayerNorm_att(XTensor& input, Model & model, py::int_ i, bool before, bool after) {
	return LayerNorm(input, model.decoder->enDeAttLayerNorms[i], model.decoder->preNorm, before, after);
}
XTensor Transformer_py::EncLayerNorm_fnn(XTensor& input, Model & model, py::int_ i, bool before, bool after) {
	return LayerNorm(input, model.encoder->fnnLayerNorms[i], model.encoder->preNorm, before, after);
}
XTensor Transformer_py::DecLayerNorm_fnn(XTensor& input, Model & model, py::int_ i, bool before, bool after) {
	return LayerNorm(input, model.decoder->fnnLayerNorms[i], model.decoder->preNorm, before, after);
}
XTensor Transformer_py::LayerNorm_encoder(XTensor& input, Model & model) {
	return model.encoder->encoderLayerNorm->Make(input);
}
XTensor Transformer_py::LayerNorm_decoder(XTensor& input, Model & model) {
	return model.decoder->decoderLayerNorm->Make(input);
}
XTensor Transformer_py::GetMask(Model &model,XTensor & padding) {
	if (model.isLM) {
		int len = padding.GetDim(padding.order - 1);
		int* dims = new int[padding.order + 2];
		for (int i = 0; i < padding.order; i++)
			dims[i + 1] = padding.GetDim(i);
		dims[0] = model.nhead;
		dims[padding.order + 1] = len;
		XTensor mask;
		InitTensor(&mask, padding.order + 2, dims, X_FLOAT, padding.devID);

		delete[] dims;

		/* a upper triangular matrix where the cells of the upper triangular are set to -1e-9.
		this matrix can be used to prevent the attention to current or following words in
		a given sequence. */
		_SetDataLowTri(&mask, 1e9F, 0);
		ScaleAndShiftMe(mask, 1.0F, -1e9F);
		return mask;
	}
	else if (model.isMT) {
		XTensor maskEnc;
		model.MakeMTMaskEnc(padding, maskEnc);
		return maskEnc;
	}

	
}
XTensor Transformer_py::EncoderAttn(Model & model,XTensor& k, XTensor& q, XTensor& v, XTensor* mask, py::bool_ isTraining,py::int_ i) {
	return model.encoder->selfAtt[i].Make(k, q, v, mask, isTraining, NULL, 1);
}
XTensor Transformer_py::DecoderAttn(Model & model, XTensor& k, XTensor& q, XTensor& v, XTensor* mask, py::bool_ isTraining, py::int_ i) {
	return model.decoder->selfAtt[i].Make(k, q, v, mask, isTraining, &model.decoder->selfAttCache[i], 1);
}
XTensor Transformer_py::EnDecoderAttn(Model & model, XTensor& k, XTensor& q, XTensor& v, XTensor* mask, py::bool_ isTraining, py::int_ i) {
	return model.decoder->enDeAtt[i].Make(k, q, v, mask, isTraining, &model.decoder->enDeAttCache[i], 2);
}
XTensor Transformer_py::Dropout_trans(const XTensor &x,py::float_ dropoutp) {
	return Dropout(x, dropoutp);
}
py::float_ Transformer_py::GetEncoderDropoutP(Model * model) {
	py::float_ dropoutp = model->encoder->dropoutP;
	return dropoutp;
}
py::float_ Transformer_py::GetDecoderDropoutP(Model * model) {
	py::float_ dropoutp = model->decoder->dropoutP;
	return dropoutp;
}
XTensor Transformer_py::FNNMake_En(Model & model, py::int_ i, XTensor& input, bool isTraining) {
	return model.encoder->fnns[i].Make(input, isTraining);
}
XTensor Transformer_py::FNNMake_De(Model & model, py::int_ i, XTensor& input, bool isTraining) {
	return model.decoder->fnns[i].Make(input, isTraining);
}
void Transformer_py::MakeCacheDisable(Model & model) {
	for (int i = 0; i < model.decoder->nlayer; i++) {
		model.decoder->selfAttCache[i].enable = false;
		model.decoder->enDeAttCache[i].enable = false;
	}
}
void Transformer_py::MakeOutputLayer(Model & model, XTensor& input, XTensor& output, bool isTraining, bool normalized) {
	model.outputLayer->Make(input, output, isTraining, normalized);
}
XTensor Transformer_py::IndexToOnehot_py(Trainer & trainer, const XTensor & index) {
	return IndexToOnehot(index,trainer.vSizeTgt,trainer.labelSmoothingP);
}
int Transformer_py::LoadConfigFromFile(Config & config, py::str configFN_s, py::list args) {
	char *arg_c[1024];
	int n = 0;
	for (auto item : args) {
		std::string s = py::cast<py::str>(item);
		arg_c[n] = new char[s.length() + 1];
		strcpy(arg_c[n], s.c_str());
		arg_c[n][s.length()] = '\0';
		n += 1;
	}
	char configFN[1024];
	std::string ss = configFN_s;
	strcpy(configFN, ss.c_str());
	configFN[ss.length()] = '\0';
	return config.LoadFromFile(configFN, arg_c);
}
void Transformer_py::Translate_Fun(Config & config) {
	DISABLE_GRAD;
	Model model;
	model.InitModel(config);
	Translator translator;
	translator.Init(config);
	translator.Translate(config.testFN, config.srcVocabFN,
		config.tgtVocabFN, config.outputFN, &model);
}

Config Transformer_py::ConfigInit(int argc, py::list argv) {
	char*arg[200];
	int n = 0;
	for (auto item : argv) {
		std::string s = py::cast<py::str>(item);
		arg[n] = new char[s.length() + 1];
		strcpy(arg[n], s.c_str());
		arg[n][s.length()] = '\0';
		n += 1;
	}
	int argc_c = argc;
	Config cc=Config(argc_c, arg);
	return cc;
}

py::float_ ReduceSumAllValue_Py(const XTensor & source) {
	float lossBatch = ReduceSumAllValue(source);
	return lossBatch;
}
py::bool_ Check_doUpdate(py::float_ lossLocal) {
	float loss = lossLocal;
	bool doUpdate = (!IsNAN(loss) && !IsINF(loss) && loss < 1e3F);
	return doUpdate;
}


int LoadConfigFromFile(Config & config,py::str configFN_s,py::list args) {
	char *arg_c[1024];
	int n = 0;
	for (auto item : args) {
		std::string s = py::cast<py::str>(item);
		arg_c[n] = new char[s.length() + 1];
		strcpy(arg_c[n], s.c_str());
		arg_c[n][s.length()] = '\0';
		n += 1;
	}
	char configFN[1024];
	std::string ss = configFN_s;
	strcpy(configFN, ss.c_str());
	configFN[ss.length()] = '\0';
	return config.LoadFromFile(configFN, arg_c);
}
void Transformer_py::Init_batchLoader(Trainer & trainer, py::str trainfn, py::bool_ training) {
	char trainc[1024];
	std::string s = trainfn;
	strcpy(trainc, s.c_str());
	trainc[s.length()] = '\0';
	trainer.batchLoader.Init(trainc, trainer.bucketSize, training);
}
py::tuple Transformer_py::LoadBatch(Trainer & trainer, XTensor* batchEnc, XTensor* paddingEnc,
	XTensor* batchDec, XTensor* paddingDec, XTensor* label, py::int_ minSentBatch, py::int_ batchSize, py::int_ devID) {
	UInt64List info = trainer.batchLoader.LoadBatch(batchEnc, paddingEnc, batchDec, paddingDec, label, minSentBatch, batchSize, devID);
	py::int_ wc = (int)info[0];
	py::int_ ws = (int)info[1];
	return py::make_tuple(wc, ws);
}
XTensor MatrixMul_py(const XTensor &a, const XTensor &b) {
	return MatrixMul(a, b);
}
void Transformer_py::MakeCheckpoint(Trainer & trainer, Config & config, Model* model, py::str ss, int id) {
	char cc[1024];
	std::string s = ss;
	strcpy(cc, s.c_str());
	cc[s.length()] = '\0';
	trainer.MakeCheckpoint(model, config.validFN, config.modelFN, cc, id);
}




void init_transformer(py::module & m) {
	py::class_<Config>(m, "Config")
		.def(py::init<>());
	py::class_<Model>(m, "Model")
		.def_readwrite("devID", &Model::devID)
		.def_readwrite("encoder", &Model::encoder)
		.def_readwrite("decoder", &Model::decoder)
		.def_readwrite("outputLayer", &Model::outputLayer)
		.def_readwrite("isLM", &Model::isLM)
		.def_readwrite("isMT", &Model::isMT)
		.def_readwrite("useFP16", &Model::useFP16)
		.def_readwrite("nhead", &Model::nhead)
		.def_readwrite("shareAllEmbeddings", &Model::shareAllEmbeddings)
		.def_readwrite("shareDecInputOutputWeight", &Model::shareDecInputOutputWeight)
		.def(py::init<>())
		.def("InitModel", &Model::InitModel)
		.def("ShowModelConfig", &Model::ShowModelConfig)
		.def("MakeEncoder", &Model::MakeEncoder, py::return_value_policy::take_ownership)
		.def("MakeDecoder", &Model::MakeDecoder, py::return_value_policy::take_ownership)
		.def("MakeLM", &Model::MakeLM)
		.def("MakeMT", &Model::MakeMT)
		.def("MakeMTMask", &Model::MakeMTMask)
		.def("MakeMTMaskEnc", &Model::MakeMTMaskEnc)
		.def("MakeMTMaskDec", &Model::MakeMTMaskDec)
		.def("GetParams", &Model::GetParams)
		.def("Dump", [](Model * model, py::str modelFN) {
		char cc[1024];
		std::string s = modelFN;
		strcpy(cc, s.c_str());
		cc[s.length()] = '\0';
		model->Dump(cc);
	})
		.def("Read", &Model::Read);
	py::class_<Embedder>(m, "Embedder")
		.def_readwrite("devID", &Embedder::devID)
		.def_readwrite("vSize", &Embedder::vSize)
		.def_readwrite("eSize", &Embedder::eSize)
		.def_readwrite("maxLength", &Embedder::maxLength)
		.def_readwrite("d", &Embedder::d)
		.def_readwrite("padIdx", &Embedder::padIdx)
		.def_readwrite("w", &Embedder::w)
		.def_readwrite("posEmbeddingBase", &Embedder::posEmbeddingBase)
		.def(py::init<>())
		.def("InitModel", &Embedder::InitModel)
		.def("MakePosEmbedding", &Embedder::MakePosEmbedding)
		.def("Make", &Embedder::Make, py::return_value_policy::take_ownership);
	py::class_<AttEncoder>(m, "AttEncoder")
		.def_readwrite("devID", &AttEncoder::devID)
		.def_readwrite("nlayer", &AttEncoder::nlayer)
		.def_readwrite("hSize", &AttEncoder::hSize)
		.def_readwrite("eSize", &AttEncoder::eSize)
		.def_readwrite("vSize", &AttEncoder::vSize)
		.def_readwrite("embedder", &AttEncoder::embedder)
		.def_readwrite("preNorm", &AttEncoder::preNorm)
		.def(py::init<>())
		.def("InitModel", &AttEncoder::InitModel)
		.def("Make", (XTensor(AttEncoder::*)(XTensor&, XTensor*, XTensor&, bool))&Attention::Make, py::return_value_policy::take_ownership)
		.def("MakeFast", &AttEncoder::MakeFast)
		.def("Make", (XTensor(AttEncoder::*)(XTensor&, XTensor*, bool))&Attention::Make, py::return_value_policy::take_ownership);
	py::class_<AttDecoder>(m, "AttDecoder")
		.def_readwrite("devID", &AttDecoder::devID)
		.def_readwrite("nlayer", &AttDecoder::nlayer)
		.def_readwrite("hSize", &AttDecoder::hSize)
		.def_readwrite("eSize", &AttDecoder::eSize)
		.def_readwrite("vSize", &AttDecoder::vSize)
		.def_readwrite("embedder", &AttDecoder::embedder)
		.def_readwrite("preNorm", &AttDecoder::preNorm)
		.def(py::init<>())
		.def("InitModel", &AttDecoder::InitModel)
		.def("Make", &AttDecoder::Make, py::return_value_policy::take_ownership)
		.def("MakeFast", &AttDecoder::MakeFast, py::return_value_policy::take_ownership);
	
	
	py::class_<TrainDataSet>(m, "TrainDataSet")
		.def_readwrite("isTraining", &TrainDataSet::isTraining)
		.def(py::init<>())
		.def("IsEmpty", &TrainDataSet::IsEmpty)
		.def("ClearBuf", &TrainDataSet::ClearBuf)
		.def("BuildBucket", &TrainDataSet::BuildBucket)
		.def("SortByLength", &TrainDataSet::SortByLength)
		.def("SortBucket", &TrainDataSet::SortBucket)
		.def("SortInBucket", &TrainDataSet::SortInBucket)
		.def("LoadDataToBuffer", &TrainDataSet::LoadDataToBuffer);
		
	py::class_<Trainer>(m, "Trainer")
		.def_readwrite("d", &Trainer::d)
		.def_readwrite("nwarmup", &Trainer::nwarmup)
		.def_readwrite("vSize", &Trainer::vSize)
		.def_readwrite("vSizeTgt", &Trainer::vSizeTgt)
		.def_readwrite("lrate", &Trainer::lrate)
		.def_readwrite("lrbias", &Trainer::lrbias)
		.def_readwrite("sBatchSize", &Trainer::sBatchSize)
		.def_readwrite("wBatchSize", &Trainer::wBatchSize)
		.def_readwrite("bucketSize", &Trainer::bucketSize)
		.def_readwrite("nepoch", &Trainer::nepoch)
		.def_readwrite("nstep", &Trainer::nstep)
		.def_readwrite("maxCheckpoint", &Trainer::maxCheckpoint)
		.def_readwrite("useAdam", &Trainer::useAdam)
		.def_readwrite("batchLoader",&Trainer::batchLoader)
		.def_readwrite("nStepCheckpoint",&Trainer::nStepCheckpoint)
		.def_readwrite("useEpochCheckpoint",&Trainer::useEpochCheckpoint)
		.def_readwrite("updateStep",&Trainer::updateStep)
		.def_readwrite("isLenSorted",&Trainer::isLenSorted)
		.def(py::init<>())
		.def("Init", &Trainer::Init)
		.def("Update",&Trainer::Update)
		.def("Train", [](Trainer & trainer, Config & config, Model & model) {
		trainer.Train(config.trainFN, config.validFN, config.modelFN, &model);
	})
		.def("Validate", &Trainer::Validate)				//此处参数
		
		.def("Update", &Trainer::Update)					//此处参数
		.def("PrepareModel", &Trainer::PrepareModel);
	py::class_<Translator>(m, "Translator")
		.def_readwrite("vSize", &Translator::vSize)
		.def_readwrite("vSizeTgt", &Translator::vSizeTgt)
		.def_readwrite("sentBatch", &Translator::sentBatch)
		.def_readwrite("wordBatch", &Translator::wordBatch)
		.def_readwrite("beamSize", &Translator::beamSize)
		.def(py::init<>())
		.def("Init", &Translator::Init)
		.def("Translate", &Translator::Translate)
		.def("Dump", &Translator::Dump);
	py::class_<Attention>(m, "Attention")
		.def_readwrite("devID", &Attention::devID)
		.def_readwrite("nhead", &Attention::nhead)
		.def_readwrite("weightQ", &Attention::weightQ)
		.def_readwrite("biasQ", &Attention::biasQ)
		.def_readwrite("weightK", &Attention::weightK)
		.def_readwrite("biasK", &Attention::biasK)
		.def_readwrite("weightV", &Attention::weightV)
		.def_readwrite("biasV", &Attention::biasV)
		.def_readwrite("wBig", &Attention::wBig)
		.def_readwrite("bBig", &Attention::bBig)
		.def_readwrite("RPEmbK", &Attention::RPEmbK)
		.def_readwrite("weightO", &Attention::weightO)
		.def_readwrite("biasO", &Attention::biasO)
		.def_readwrite("dk", &Attention::dk)
		.def_readwrite("dv", &Attention::dv)
		.def_readwrite("d", &Attention::d)
		.def_readwrite("useRPR", &Attention::useRPR)
		.def_readwrite("maxRP", &Attention::maxRP)
		.def(py::init<>())
		.def("InitModel", &Attention::InitModel)
		.def("Make", &Attention::Make, py::return_value_policy::take_ownership)
		.def("MakeAttention", &Attention::MakeAttention, py::return_value_policy::take_ownership)
		.def("MakeRPRAttention", &Attention::MakeRPRAttention, py::return_value_policy::take_ownership)
		.def("GetRPEmbedding", &Attention::GetRPEmbedding, py::return_value_policy::take_ownership)
		.def("RPDotProduct", &Attention::RPDotProduct, py::return_value_policy::take_ownership);
	py::class_<Transformer_py>(m, "Transformer_py")
		.def(py::init<>())
		.def("ConfigInit", &Transformer_py::ConfigInit, py::return_value_policy::take_ownership)
		.def("Config_dict", &Transformer_py::Config_dict, py::return_value_policy::take_ownership)
		.def("Cache_disable", &Transformer_py::Cache_disable)
		.def("Embedding_En", &Transformer_py::PyEmbedding_En, py::arg("model"), py::arg("input"), py::arg("isDec"), py::arg("isTraining"), py::arg("nstep") = 0, py::return_value_policy::take_ownership)
		.def("Embedding_De", &Transformer_py::PyEmbedding_De, py::arg("model"), py::arg("input"), py::arg("isDec"), py::arg("isTraining"), py::arg("nstep") = 0, py::return_value_policy::take_ownership)
		.def("Dropout", &Transformer_py::Dropout_trans, py::return_value_policy::take_ownership)
		.def("GetEncoderLayers", &Transformer_py::GetEncoderLayers, py::return_value_policy::take_ownership)
		.def("GetDecoderLayers", &Transformer_py::GetDecoderLayers, py::return_value_policy::take_ownership)
		.def("EncLayerNorm_att", &Transformer_py::EncLayerNorm_att, py::return_value_policy::take_ownership)
		.def("DecLayerNorm_att", &Transformer_py::DecLayerNorm_att, py::return_value_policy::take_ownership)
		.def("EnDecLayerNorm_att", &Transformer_py::EnDecLayerNorm_att, py::return_value_policy::take_ownership)
		.def("EncLayerNorm_fnn", &Transformer_py::EncLayerNorm_fnn, py::return_value_policy::take_ownership)
		.def("DecLayerNorm_fnn", &Transformer_py::DecLayerNorm_fnn, py::return_value_policy::take_ownership)
		.def("GetMask", &Transformer_py::GetMask, py::return_value_policy::take_ownership)
		.def("EncoderAttn", &Transformer_py::EncoderAttn, py::return_value_policy::take_ownership)
		.def("DecoderAttn", &Transformer_py::DecoderAttn, py::return_value_policy::take_ownership)
		.def("EnDecoderAttn", &Transformer_py::EnDecoderAttn, py::return_value_policy::take_ownership)
		.def("GetEncoderDropoutP", &Transformer_py::GetEncoderDropoutP, py::return_value_policy::take_ownership)		//需要修改
		.def("GetDecoderDropoutP", &Transformer_py::GetDecoderDropoutP, py::return_value_policy::take_ownership)
		.def("FNNMake_En", &Transformer_py::FNNMake_En, py::return_value_policy::take_ownership)
		.def("FNNMake_De", &Transformer_py::FNNMake_De, py::return_value_policy::take_ownership)
		.def("LayerNorm_encoder", &Transformer_py::LayerNorm_encoder, py::return_value_policy::take_ownership)
		.def("LayerNorm_decoder", &Transformer_py::LayerNorm_decoder, py::return_value_policy::take_ownership)
		.def("MakeCacheDisable", &Transformer_py::MakeCacheDisable)
		.def("MakeOutputLayer", &Transformer_py::MakeOutputLayer)
		.def("IndexToOnehot", &Transformer_py::IndexToOnehot_py, py::return_value_policy::take_ownership)
		.def("Dump", &Transformer_py::ModelDump)
		.def("LoadConfigFromFile", &Transformer_py::LoadConfigFromFile, py::return_value_policy::take_ownership)
		.def("Translate_Fun", &Transformer_py::Translate_Fun)
		.def("Init_batchLoader", &Transformer_py::Init_batchLoader)
		.def("LoadBatch", &Transformer_py::LoadBatch, py::return_value_policy::take_ownership)
		.def("MakeCheckpoint", &Transformer_py::MakeCheckpoint)
		.def("NMTMain", &Transformer_py::NMTMain_py);
	m.def("CrossEntropy_trans", (XTensor (*)(const XTensor &, const XTensor &, const XTensor &, int))&CrossEntropy, py::arg("output"), py::arg("gold"), py::arg("padding"), py::arg("leadingDim") = -1, py::return_value_policy::take_ownership);
	//m.def("CrossEntropy_trans", py::overload_cast<const XTensor &, const XTensor &, const XTensor &, int>(&CrossEntropy), py::arg("output"), py::arg("gold"), py::arg("padding"), py::arg("leadingDim")=-1,py::return_value_policy::take_ownership);
	m.def("Sum", (XTensor(*)(const XTensor &, const XTensor &, bool, DTYPE))&Sum, py::arg("a"), py::arg("b"), py::arg("inplace") = false, py::arg("beta") = 1.0, py::return_value_policy::take_ownership);
	//m.def("Sum", py::overload_cast<const XTensor &, const XTensor &, bool, DTYPE>(&Sum), py::arg("a"), py::arg("b"), py::arg("inplace")=false, py::arg("beta")=1.0,py::return_value_policy::take_ownership);
	m.def("ReduceSumAllValue", &ReduceSumAllValue_Py, py::return_value_policy::take_ownership);
	m.def("Check_doUpdate", &Check_doUpdate, py::return_value_policy::take_ownership);
	m.def("Split",(XTensor(*)(const XTensor &, int , int )) &Split,py::return_value_policy::take_ownership);
	m.def("MatrixMul", &MatrixMul_py, py::return_value_policy::take_ownership);

}
