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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2019-03-27
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04, 2020-06
 */

#include "Search.h"
#include "Translator.h"
#include "../Utility.h"
#include "../../../tensor/XTensor.h"
#include "../../../tensor/XUtility.h"
#include "../../../tensor/core/CHeader.h"

using namespace nts;

namespace nmt
{

/* constructor */
Translator::Translator()
{
}

/* de-constructor */
Translator::~Translator()
{
    if (beamSize > 1)
        delete (BeamSearch*)seacher;
    else
        delete (GreedySearch*)seacher;
}

/* initialize the model */
void Translator::Init(Config& config)
{
    beamSize = config.beamSize;
    vSize = config.srcVocabSize;
    vSizeTgt = config.tgtVocabSize;
    sentBatch = config.sBatchSize;
    wordBatch = config.wBatchSize;

    if (beamSize > 1) {
        LOG("translating with beam search (%d)", beamSize);
        seacher = new BeamSearch();
        ((BeamSearch*)seacher)->Init(config);
    }
    else if (beamSize == 1) {
        LOG("translating with greedy search");
        seacher = new GreedySearch();
        ((GreedySearch*)seacher)->Init(config);
    }
    else {
        CheckNTErrors(false, "Invalid beam size\n");
    }
}

/*
test the model
>> ifn - input data file
>> sfn - source vocab file
>> tfn - target vocab file
>> ofn - output data file
>> model - pretrained model
*/
void Translator::Translate(const char* ifn, const char* sfn, 
                           const char* tfn, const char* ofn, Model* model)
{
    int wc = 0;
    int wordCountTotal = 0;
    int sentCount = 0;
    int batchCount = 0;

    int devID = model->devID;

    double startT = GetClockSec();

    /* batch of input sequences */
    XTensor batchEnc;

    /* padding */
    XTensor paddingEnc;

    batchLoader.Init(ifn, sfn, tfn);
    LOG("loaded the input file, elapsed=%.1fs ", GetClockSec() - startT);

    int count = 0;
    double batchStart = GetClockSec();
    while (!batchLoader.IsEmpty())
    {
        count++;

        for (int i = 0; i < model->decoder->nlayer; ++i) {
            model->decoder->selfAttCache[i].miss = true;
            model->decoder->enDeAttCache[i].miss = true;
        }

        auto indices = batchLoader.LoadBatch(&batchEnc, &paddingEnc, 
                                             sentBatch, wordBatch, devID);

        IntList* output = new IntList[indices.Size() - 1];

        /* greedy search */
        if (beamSize == 1) {
            ((GreedySearch*)seacher)->Search(model, batchEnc, paddingEnc, output);
        }
        /* beam search */
        else {
            XTensor score;
            ((BeamSearch*)seacher)->Search(model, batchEnc, paddingEnc, output, score);
        }

        for (int i = 0; i < indices.Size() - 1; ++i) {
            Result* res = new Result;
            res->id = int(indices[i]);
            res->res = output[i];
            batchLoader.outputBuffer.Add(res);
        }
        delete[] output;

        wc += int(indices[-1]);
        wordCountTotal += int(indices[-1]);

        sentCount += int(indices.Size() - 1);
        batchCount += 1;

        if (count % 1 == 0) {
            double elapsed = GetClockSec() - batchStart;
            batchStart = GetClockSec();
            LOG("elapsed=%.1fs, sentence=%f, sword=%.1fw/s",
                elapsed, float(sentCount) / float(batchLoader.inputBuffer.Size()), 
                double(wc) / elapsed);
            wc = 0;
        }
    }

    /* append empty lines to the result */
    for (int i = 0; i < batchLoader.emptyLines.Size(); i++) {
        Result* emptyRes = new Result;
        emptyRes->id = batchLoader.emptyLines[i];
        batchLoader.outputBuffer.Add(emptyRes);
    }

    //double startDump = GetClockSec();

    /* reorder the result */
    batchLoader.SortOutput();

    /* print the result to a file */
    batchLoader.DumpRes(ofn);

    //double elapsed = GetClockSec() - startDump;

    LOG("translation completed (word=%d, sent=%zu)", 
        wordCountTotal, batchLoader.inputBuffer.Size() + batchLoader.emptyLines.Size());
}

/*
dump the result into the file
>> file - data file
>> output - output tensor
*/
void Translator::Dump(FILE* file, XTensor* output)
{
    if (output != NULL && output->unitNum != 0) {
        int seqLength = output->dimSize[output->order - 1];

        for (int i = 0; i < output->unitNum; i += seqLength) {
            for (int j = 0; j < seqLength; j++) {
                int w = output->GetInt(i + j);
                if (w < 0 || w == 1 || w == 2)
                    break;
                fprintf(file, "%d ", w);
            }

            fprintf(file, "\n");
        }
    }
    else
    {
        fprintf(file, "\n");
    }
}

}
