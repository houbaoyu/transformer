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
 * $Created by: HU Chi (huchinlp@foxmail.com) 2019-04-03
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-06
 */

#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <algorithm>

#include "DataSet.h"
#include "../Utility.h"

using namespace nmt;

namespace nts {

/* sort the output by id (in ascending order) */
void DataSet::SortInput() {
    sort(inputBuffer.items, inputBuffer.items + inputBuffer.count, 
        [](Example* a, Example* b) {
            return a->values.count > b->values.count;
        });
}

/* sort the input by length (in descending order) */
void DataSet::SortOutput() {
    sort(outputBuffer.items, outputBuffer.items + outputBuffer.count, 
        [](Result* a, Result* b) {
            return a->id < b->id;
        });
}

/*
load data from the file to the buffer
*/
void DataSet::LoadDataToBuffer()
{
    string line;
    inputBuffer.Clear();
    bufferUsed = 0;

    int id = 0;
    const string tokenDelimiter = " ";

    while (getline(*fp, line)) {
        IntList values;

        /* load words and transform them to ids */
        auto indices = SplitToPos(line, tokenDelimiter);

        /* reserve the first 120 words if the input is too long */
        size_t maxLen = indices.Size() > MAX_WORD_NUM ? MAX_WORD_NUM : indices.Size();

        for (size_t i = 0; i < maxLen; i++) {
            size_t offset = (i != (indices.Size() - 1)) ?
                              (size_t)indices[(int)i + 1] - (size_t)indices[(int)i] - tokenDelimiter.size()
                            : line.size() - (size_t)indices[(int)i];
            string word = line.substr((size_t)indices[(int)i], offset);
            if (srcVocab.word2id.find(word) == srcVocab.word2id.end())
                values.Add(UNK);
            else
                values.Add(srcVocab.word2id.at(word));
        }

        /* make sure that the sequence ends with EOS */
        if (values.Size() != 0 && values[-1] != EOS)
            values.Add(EOS);

        Example* example = new Example;
        example->id = id;
        example->values = values;
        if (values.Size() != 0)
            inputBuffer.Add(example);
        else
            emptyLines.Add(id);
        id++;
    }
    fp->close();

    SortInput();

    XPRINT1(0, stderr, "[INFO] loaded %d sentences\n", id);
}

/*
load a mini-batch to the device (for translating)
>> batchEnc - a tensor to store the batch of input
>> paddingEnc - a tensor to store the batch of paddings
>> minSentBatch - the minimum number of sentence batch
>> batchSize - the maxium number of words in a batch
>> devID - the device id, -1 for the CPU
<< indices of the sentences
*/
UInt64List DataSet::LoadBatch(XTensor* batchEnc, XTensor* paddingEnc,
                              int minSentBatch, int batchSize, int devID)
{
    int realBatchSize = minSentBatch;

    /* get the maximum sentence length in a mini-batch */
    int maxLen = (int)inputBuffer[(int)bufferUsed]->values.Size();

    /* dynamic batching for sentences */
    //while ((realBatchSize < (inputBuffer.Size() - bufferUsed))
    //    && (realBatchSize * maxLen < batchSize)) {
    //    realBatchSize++;
    //}

    /* real batch size */
    if ((inputBuffer.Size() - bufferUsed) < realBatchSize) {
        realBatchSize = (int)(inputBuffer.Size() - bufferUsed);
    }

    CheckNTErrors(maxLen != 0, "invalid length");

    int* batchValues = new int[realBatchSize * maxLen];
    float* paddingValues = new float[realBatchSize * maxLen];

    for (int i = 0; i < realBatchSize * maxLen; i++) {
        batchValues[i] = PAD;
        paddingValues[i] = 1.0F;
    }

    size_t curSrc = 0;

    /* right padding */
    UInt64List infos;
    size_t totalLength = 0;

    for (size_t i = 0; i < (size_t)realBatchSize; ++i) {
        infos.Add(inputBuffer[(int)(bufferUsed + i)]->id);
        totalLength += inputBuffer[(int)(bufferUsed + i)]->values.Size();

        curSrc = maxLen * i;
        for (size_t j = 0; j < inputBuffer[(int)(bufferUsed + i)]->values.Size(); j++)
            batchValues[(int)(curSrc++)] = (int)inputBuffer[(int)(bufferUsed + i)]->values[(int)j];
        while (curSrc < maxLen * (i + 1))
            paddingValues[(int)(curSrc++)] = 0;
    }
    infos.Add(totalLength);

    InitTensor2D(batchEnc, realBatchSize, maxLen, X_INT, devID);
    InitTensor2D(paddingEnc, realBatchSize, maxLen, X_FLOAT, devID);

    bufferUsed += realBatchSize;

    batchEnc->SetData(batchValues, batchEnc->unitNum);
    paddingEnc->SetData(paddingValues, paddingEnc->unitNum);

    delete[] batchValues;
    delete[] paddingValues;

    return infos;
}

/*
the constructor of DataSet
>> dataFile - path of the data file
>> srcVocabFN - path of the source vocab file
>> tgtVocabFN - path of the target vocab file
*/
void DataSet::Init(const char* dataFile, const char* srcVocabFN, const char* tgtVocabFN)
{
    fp = new ifstream(dataFile);
    CheckNTErrors(fp->is_open(), "Can not open the test data");
    bufferUsed = 0;

    CheckNTErrors(strcmp(srcVocabFN, "") != 0, "missing source vocab file");
    CheckNTErrors(strcmp(tgtVocabFN, "") != 0, "missing target vocab file");

    srcVocab.Load(srcVocabFN);

    /* share source and target vocabs */
    if (strcmp(srcVocabFN, tgtVocabFN) == 0) {
        XPRINT(0, stderr, "[INFO] share source and target vocabs \n");
        tgtVocab.CopyFrom(srcVocab);
    }
    else {
        tgtVocab.Load(tgtVocabFN);
    }

    LoadDataToBuffer();
}

/* check if the buffer is empty */
bool DataSet::IsEmpty() {
    if (bufferUsed < inputBuffer.Size())
        return false;
    return true;
}

/* dump the translation to a file */
void DataSet::DumpRes(const char* ofn)
{
    ofstream ofile(ofn, ios::out);

    for (int t = 0; t < outputBuffer.Size(); t++) {
        auto res = outputBuffer[t];
        for (int i = 0; i < res->res.Size(); i++) {
            if (res->res[i] < 4)
                break;
            ofile << tgtVocab.id2word[res->res[i]] << " ";
        }
        ofile << "\n";
    }

    ofile.close();
}

/* de-constructor */
DataSet::~DataSet()
{
    /* release the file */
    delete fp;

    /* release the input buffer */
    for (int i = 0; i < inputBuffer.Size(); i++)
        delete inputBuffer[i];

    /* release the output buffer */
    for (int i = 0; i < outputBuffer.Size(); i++)
        delete outputBuffer[i];
}

}