/* NiuTrans.NMT - an open-source neural machine translation system.
 * Copyright (C) 2020 NiuTrans Research. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04, 2020-06
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>

#include "Utility.h"
#include "../../tensor/XGlobal.h"
#include "../../tensor/XConfig.h"

using namespace nts;
using namespace std;

namespace nmt
{

/*
load configurations from the command
>> argc - number of arguments
>> argv - the list of arguments
*/
Config::Config(int argc, char** argv)
{
    char** args = new char* [MAX_PARAM_NUM];
    for (int i = 0; i < argc; i++) {
        args[i] = new char[strlen(argv[i]) + 1];
        strcpy(args[i], argv[i]);
    }

    char* configFN = new char[1024];
    LoadParamString(argc, args, "config", configFN, "");

    int argsNum = argc;

    /* load configurations from a file */
    if (strcmp(configFN, "") != 0)
        argsNum = LoadFromFile(configFN, args);

    ShowParams(argsNum, args);

    /* options for the model */
    LoadParamInt(argsNum, args, "nhead", &nhead, 4);
    LoadParamInt(argsNum, args, "enclayer", &nEncLayer, 1);
    LoadParamInt(argsNum, args, "declayer", &nDecLayer, 1);
    LoadParamInt(argsNum, args, "maxrp", &maxRP, 8);
    LoadParamInt(argsNum, args, "embsize", &embSize, 16);
    LoadParamInt(argsNum, args, "modelsize", &modelSize, 16);
    LoadParamInt(argsNum, args, "maxpos", &maxPosLen, 1024);
    LoadParamInt(argsNum, args, "fnnhidden", &fnnHiddenSize, modelSize * 2);
    LoadParamInt(argsNum, args, "vsize", &srcVocabSize, 10152);
    LoadParamInt(argsNum, args, "vsizetgt", &tgtVocabSize, 10152);
    LoadParamInt(argsNum, args, "padid", &padID, 1);
    LoadParamInt(argsNum, args, "startid", &startID, 2);
    LoadParamInt(argsNum, args, "endid", &endID, 2);
    LoadParamBool(argsNum, args, "rpr", &useRPR, false);
    LoadParamBool(argsNum, args, "prenorm", &preNorm, true);

    // TODO: refactor the parameters type to support weight sharing during training
    LoadParamInt(argsNum, args, "shareemb", &shareAllEmbeddings, 0);
    LoadParamInt(argsNum, args, "sharedec", &shareDecInputOutputWeight, 0);
    LoadParamString(argsNum, args, "model", modelFN, "");
    LoadParamString(argsNum, args, "srcvocab", srcVocabFN, "vocab.src");
    LoadParamString(argsNum, args, "tgtvocab", tgtVocabFN, "vocab.tgt");

    /* options for training */
    LoadParamString(argsNum, args, "train", trainFN, "");
    LoadParamString(argsNum, args, "valid", validFN, "");
    LoadParamInt(argsNum, args, "dev", &devID, 0);
    LoadParamInt(argsNum, args, "wbatch", &wBatchSize, 4096);
    LoadParamInt(argsNum, args, "sbatch", &sBatchSize, 8);
    isTraining = (strcmp(trainFN, "") == 0) ? false : true;
    LoadParamBool(argsNum, args, "mt", &isMT, true);
    LoadParamFloat(argsNum, args, "dropout", &dropout, 0.3F);
    LoadParamFloat(argsNum, args, "fnndrop", &fnnDropout, 0.1F);
    LoadParamFloat(argsNum, args, "attdrop", &attDropout, 0.1F);

    LoadParamFloat(argc, args, "lrate", &lrate, 0.0015F);
    LoadParamFloat(argc, args, "lrbias", &lrbias, 0);
    LoadParamInt(argc, args, "nepoch", &nepoch, 50);
    LoadParamInt(argc, args, "maxcheckpoint", &maxCheckpoint, 10);
    LoadParamInt(argc, args, "nstep", &nstep, 100000);
    LoadParamInt(argc, args, "nwarmup", &nwarmup, 8000);
    LoadParamBool(argc, args, "adam", &useAdam, true);
    LoadParamFloat(argc, args, "adambeta1", &adamBeta1, 0.9F);
    LoadParamFloat(argc, args, "adambeta2", &adamBeta2, 0.98F);
    LoadParamFloat(argc, args, "adamdelta", &adamDelta, 1e-9F);
    LoadParamBool(argc, args, "shuffled", &isShuffled, false);
    LoadParamFloat(argc, args, "labelsmoothing", &labelSmoothingP, 0.1F);
    LoadParamInt(argc, args, "nstepcheckpoint", &nStepCheckpoint, -1);
    LoadParamBool(argc, args, "epochcheckpoint", &useEpochCheckpoint, true);
    LoadParamInt(argc, args, "updatestep", &updateStep, 1);
    LoadParamBool(argc, args, "sorted", &isLenSorted, false);

    LoadParamInt(argc, args, "bufsize", &bufSize, 50000);
    LoadParamBool(argc, args, "doubledend", &isDoubledEnd, false);
    LoadParamBool(argc, args, "smallbatch", &isSmallBatch, true);
    LoadParamBool(argc, args, "bigbatch", &isBigBatch, false);
    LoadParamBool(argc, args, "randbatch", &isRandomBatch, false);
    LoadParamInt(argc, args, "bucketsize", &bucketSize, wBatchSize * 10);

    /* options for translating */
    LoadParamString(argsNum, args, "test", testFN, "");
    LoadParamString(argsNum, args, "output", outputFN, "");
    LoadParamInt(argsNum, args, "beamsize", &beamSize, 1);
    LoadParamBool(argsNum, args, "fp16", &useFP16, false);
    LoadParamFloat(argsNum, args, "lenalpha", &lenAlpha, 0.6F);
    LoadParamFloat(argsNum, args, "maxlenalpha", &maxLenAlpha, 1.2F);

    for (int i = 0; i < argc; i++)
        delete[] args[i];
    delete[] args;
    delete[] configFN;
}

/*
load configurations from a file
>> configFN - path to the configuration file
>> args - the list to store the configurations
format: one option per line, separated by a blank or a tab
*/
int Config::LoadFromFile(const char* configFN, char** args) {
    ifstream f(configFN, ios::in);
    CheckNTErrors(f.is_open(), "unable to open the config file");

    int argsNum = 0;

    /* parse arguments */
    string key, value;
    while (f >> key >> value) {
        key += '-';
        strcpy(args[argsNum++], key.c_str());
        strcpy(args[argsNum++], value.c_str());
    }

    /* record the number of arguments */
    return argsNum;
}

#define MAX_WORD_NUM 120

/*
split string by delimiter, this will return indices of all sub-strings
>> s - the original string
>> delimiter - as it is
<< indices - indices of all sub-strings
*/
UInt64List SplitToPos(const string& s, const string& delimiter)
{
    UInt64List indices;
    if (delimiter.length() == 0) {
        indices.Add(0);
    }
    size_t pos = 0;
    uint64_t start = 0;
    while ((pos = s.find(delimiter, start)) != string::npos) {
        if (pos != start) {
            indices.Add(start);
        }
        start = pos + delimiter.length();
    }
    if (start != s.length()) {
        indices.Add(start);
    }
    return indices;
}

/* split a string to a int64_t list */
IntList SplitInt(const string& s, const string& delimiter)
{
    IntList values;
    auto indices = SplitToPos(s, delimiter);
    for (int i = 0; i < indices.Size(); i++) {
        
        /* this line is with problem. Why do we need an IntList to keep an int64*/
        values.Add((int)strtol(s.data() + indices[i], nullptr, 10));
    }
    return values;
}

/* split a string to a float list */
FloatList SplitFloat(const string& s, const string& delimiter)
{
    FloatList values;
    auto indices = SplitToPos(s, delimiter);
    for (int i = 0; i < indices.Size(); i++) {
        values.Add(strtof(s.data() + indices[i], nullptr));
    }
    return values;
}

}
