﻿/* NiuTrans.NMT - an open-source neural machine translation system.
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
 * $Created by: HU Chi (huchinlp@foxmail.com) 2020-08-09
 * TODO: refactor the data loader class and references
 */

#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <algorithm>

#include "TrainDataSet.h"
#include "../Utility.h"
#include "../translate/Vocab.h"

using namespace nmt;

namespace nts {

/* sort the dataset by length (in descending order) */
void TrainDataSet::SortByLength() {
    sort(buffer.items, buffer.items + buffer.count,
        [](TrainExample* a, TrainExample* b) {
            return (a->srcSent.Size() + a->tgtSent.Size())
                 > (b->srcSent.Size() + b->tgtSent.Size());
        });
}

/* sort buckets by key (in descending order) */
void TrainDataSet::SortBucket() {
    sort(buffer.items, buffer.items + buffer.count,
        [](TrainExample* a, TrainExample* b) {
            return a->bucketKey > b->bucketKey;
        });
}

/*
sort the output by key in a range (in descending order)
>> begin - the first index of the range
>> end - the last index of the range
*/
void TrainDataSet::SortInBucket(int begin, int end) {
    sort(buffer.items + begin, buffer.items + end,
        [](TrainExample* a, TrainExample* b) {
            return (a->key > b->key);
        });
}

/*
load all data from a file to the buffer
training data format (binary):
first 8 bit: number of sentence pairs
subsequent segements:
source sentence length (4 bit)
target sentence length (4 bit)
source tokens (4 bit per token)
target tokens (4 bit per token)
*/
void TrainDataSet::LoadDataToBuffer()
{
    buffer.Clear();
    curIdx = 0;

    int id = 0;
    uint64_t sentNum = 0;

    int srcVocabSize = 0;
    int tgtVocabSize = 0;
    fread(&srcVocabSize, sizeof(srcVocabSize), 1, fp);
    fread(&tgtVocabSize, sizeof(tgtVocabSize), 1, fp);

    fread(&sentNum, sizeof(uint64_t), 1, fp);
    CheckNTErrors(sentNum > 0, "Invalid sentence pairs number");

    while (id < sentNum) {
        int srcLen = 0;
        int tgtLen = 0;
        fread(&srcLen, sizeof(int), 1, fp);
        fread(&tgtLen, sizeof(int), 1, fp);
        CheckNTErrors(srcLen > 0, "Invalid source sentence length");
        CheckNTErrors(tgtLen > 0, "Invalid target sentence length");

        IntList srcSent;
        IntList tgtSent;
        srcSent.ReadFromFile(fp, srcLen);
        tgtSent.ReadFromFile(fp, tgtLen);

        TrainExample* example = new TrainExample;
        example->id = id++;
        example->key = id;
        example->srcSent = srcSent;
        example->tgtSent = tgtSent;

        buffer.Add(example);
    }

    fclose(fp);

    XPRINT1(0, stderr, "[INFO] loaded %d sentences\n", id);
}

/*
load a mini-batch to the device (for training)
>> batchEnc - a tensor to store the batch of encoder input
>> paddingEnc - a tensor to store the batch of encoder paddings
>> batchDec - a tensor to store the batch of decoder input
>> paddingDec - a tensor to store the batch of decoder paddings
>> label - a tensor to store the label of input
>> minSentBatch - the minimum number of sentence batch
>> batchSize - the maxium number of words in a batch
>> devID - the device id, -1 for the CPU
<< return - number of target tokens and sentences
*/
UInt64List TrainDataSet::LoadBatch(XTensor* batchEnc, XTensor* paddingEnc,
                                   XTensor* batchDec, XTensor* paddingDec, XTensor* label,
                                   size_t minSentBatch, size_t batchSize, int devID)
{
    UInt64List info;
    size_t srcTokenNum = 0;
    size_t tgtTokenNum = 0;
    size_t realBatchSize = 1;

    if (!isTraining)
        realBatchSize = minSentBatch;

    /* get the maximum source sentence length in a mini-batch */
    size_t maxSrcLen = buffer[(int)curIdx]->srcSent.Size();

    /* max batch size */
    const int MAX_BATCH_SIZE = 512;

    /* dynamic batching for sentences, enabled when the dataset is used for training */
    if (isTraining) {
        while ((realBatchSize < (buffer.Size() - curIdx))
            && (realBatchSize * maxSrcLen < batchSize)
            && (realBatchSize < MAX_BATCH_SIZE)
            && (realBatchSize * buffer[(int)(curIdx + realBatchSize)]->srcSent.Size() < batchSize)) {
            if (maxSrcLen < buffer[(int)(curIdx + realBatchSize)]->srcSent.Size())
                maxSrcLen = buffer[(int)(curIdx + realBatchSize)]->srcSent.Size();
            realBatchSize++;
        }
    }
    
    /* real batch size */
    if ((buffer.Size() - curIdx) < realBatchSize) {
        realBatchSize = buffer.Size() - curIdx;
    }

    CheckNTErrors(realBatchSize > 0, "Invalid batch size");

    /* get the maximum target sentence length in a mini-batch */
    size_t maxTgtLen = buffer[(int)curIdx]->tgtSent.Size();
    for (size_t i = 0; i < realBatchSize; i++) {
        if (maxTgtLen < buffer[(int)(curIdx + i)]->tgtSent.Size())
            maxTgtLen = buffer[(int)(curIdx + i)]->tgtSent.Size();
    }
    for (size_t i = 0; i < realBatchSize; i++) {
        if (maxSrcLen < buffer[(int)(curIdx + i)]->srcSent.Size())
            maxSrcLen = buffer[(int)(curIdx + i)]->srcSent.Size();
    }

    CheckNTErrors(maxSrcLen != 0, "Invalid source length for batching");

    int* batchEncValues = new int[realBatchSize * maxSrcLen];
    float* paddingEncValues = new float[realBatchSize * maxSrcLen];

    int* labelVaues = new int[realBatchSize * maxTgtLen];
    int* batchDecValues = new int[realBatchSize * maxTgtLen];
    float* paddingDecValues = new float[realBatchSize * maxTgtLen];

    for (int i = 0; i < realBatchSize * maxSrcLen; i++) {
        batchEncValues[i] = PAD;
        paddingEncValues[i] = 1;
    }
    for (int i = 0; i < realBatchSize * maxTgtLen; i++) {
        batchDecValues[i] = PAD;
        labelVaues[i] = PAD;
        paddingDecValues[i] = 1.0F;
    }

    size_t curSrc = 0;
    size_t curTgt = 0;

    /*
    batchEnc: end with EOS (left padding)
    batchDec: begin with SOS (right padding)
    label:    end with EOS (right padding)
    */
    for (int i = 0; i < realBatchSize; ++i) {

        srcTokenNum += buffer[(int)(curIdx + i)]->srcSent.Size();
        tgtTokenNum += buffer[(int)(curIdx + i)]->tgtSent.Size();

        curSrc = maxSrcLen * i;
        for (int j = 0; j < buffer[(int)(curIdx + i)]->srcSent.Size(); j++) {
            batchEncValues[curSrc++] = buffer[(int)(curIdx + i)]->srcSent[j];
        }

        curTgt = maxTgtLen * i;
        for (int j = 0; j < buffer[(int)(curIdx + i)]->tgtSent.Size(); j++) {
            if (j > 0)
                labelVaues[curTgt - 1] = buffer[(int)(curIdx + i)]->tgtSent[j];
            batchDecValues[curTgt++] = buffer[(int)(curIdx + i)]->tgtSent[j];
        }
        labelVaues[curTgt - 1] = EOS;
        while (curSrc < maxSrcLen * (i + 1))
            paddingEncValues[curSrc++] = 0;
        while (curTgt < maxTgtLen * (i + 1))
            paddingDecValues[curTgt++] = 0;

    }

    int rbs = (int)realBatchSize;
    int msl = (int)maxSrcLen;
	int mtl = (int)maxTgtLen;
    InitTensor2D(batchEnc, rbs, msl, X_INT, devID);
    InitTensor2D(paddingEnc, rbs, msl, X_FLOAT, devID);
    InitTensor2D(batchDec, rbs, mtl, X_INT, devID);
    InitTensor2D(paddingDec, rbs, mtl, X_FLOAT, devID);
    InitTensor2D(label, rbs, mtl, X_INT, devID);

    curIdx += realBatchSize;

    batchEnc->SetData(batchEncValues, batchEnc->unitNum);
    paddingEnc->SetData(paddingEncValues, paddingEnc->unitNum);
    batchDec->SetData(batchDecValues, batchDec->unitNum);
    paddingDec->SetData(paddingDecValues, paddingDec->unitNum);
    label->SetData(labelVaues, label->unitNum);

    delete[] batchEncValues;
    delete[] paddingEncValues;
    delete[] batchDecValues;
    delete[] paddingDecValues;
    delete[] labelVaues;

    info.Add(tgtTokenNum);
    info.Add(realBatchSize);
    return info;
}

/*
the constructor of DataSet
>> dataFile - path of the data file
>> bucketSize - size of the bucket to keep similar length sentence pairs
>> training - indicates whether it is used for training
*/
void TrainDataSet::Init(const char* dataFile, int myBucketSize, bool training)
{
    fp = fopen(dataFile, "rb");
    CheckNTErrors(fp, "can not open the training file");
    curIdx = 0;
    bucketSize = myBucketSize;
    isTraining = training;

    LoadDataToBuffer();

    SortByLength();

    if (isTraining)
        BuildBucket();
}

/* check if the buffer is empty */
bool TrainDataSet::IsEmpty() {
    if (curIdx < buffer.Size())
        return false;
    return true;
}

/* reset the buffer */
void TrainDataSet::ClearBuf()
{
    curIdx = 0;

    /* make different batches in different epochs */
    SortByLength();

    if (isTraining)
        BuildBucket();
}

/* group data into buckets with similar length */
void TrainDataSet::BuildBucket()
{
    size_t idx = 0;

    /* build and shuffle buckets */
    while (idx < buffer.Size()) {

        /* sentence number in a bucket */
        size_t sentNum = 1;

        /* get the maximum source sentence length in a bucket */
        size_t maxSrcLen = buffer[(int)idx]->srcSent.Size();

        /* bucketing for sentences */
        while ((sentNum < (buffer.Size() - idx))
            && (sentNum * maxSrcLen < bucketSize)
            && (sentNum * buffer[(int)(curIdx + sentNum)]->srcSent.Size() < bucketSize)) {
            if (maxSrcLen < buffer[(int)(idx + sentNum)]->srcSent.Size())
                maxSrcLen = buffer[(int)(idx + sentNum)]->srcSent.Size();
            sentNum++;
        }

        /* make sure the number is valid */
        if ((buffer.Size() - idx) < sentNum) {
            sentNum = buffer.Size() - idx;
        }

        int randomKey = rand();

        /* shuffle items in a bucket */
        for (size_t i = 0; i < sentNum; i++) {
            buffer[(int)(idx + i)]->bucketKey = randomKey;
        }

        idx += sentNum;
    }
    SortBucket();

    /* sort items in a bucket */
    idx = 0;
    while (idx < buffer.Size()) {
        size_t sentNum = 0;
        int bucketKey = buffer[(int)(idx + sentNum)]->bucketKey;
        while (sentNum < (buffer.Size() - idx)
            && buffer[(int)(idx + sentNum)]->bucketKey == bucketKey) {
            buffer[(int)(idx + sentNum)]->key = (int)buffer[(int)(idx + sentNum)]->srcSent.Size();
            sentNum++;
        }
        SortInBucket((int)idx, (int)(idx + sentNum));
        idx += sentNum;
    }
}

/* de-constructor */
TrainDataSet::~TrainDataSet()
{

    /* release the buffer */
    for (int i = 0; i < buffer.Size(); i++)
        delete buffer[i];
}

}