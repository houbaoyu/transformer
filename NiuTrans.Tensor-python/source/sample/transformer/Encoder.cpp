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
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#include "Encoder.h"
#include "Utility.h"
#include "submodel/LayerNorm.h"
#include "submodel/CommonModules.h"
#include "../../tensor/core/CHeader.h"

namespace nmt
{

/* constructor */
AttEncoder::AttEncoder()
{
    selfAtt = NULL;
    fnns = NULL;
    attLayerNorms = NULL;
    fnnLayerNorms = NULL;
    encoderLayerNorm = NULL;
}

/* de-constructor */
AttEncoder::~AttEncoder()
{
    delete[] selfAtt;
    delete[] fnns;
    delete[] attLayerNorms;
    delete[] fnnLayerNorms;
    if (preNorm)
        delete encoderLayerNorm;
}

/*
initialize the model
>> config - configurations for the model
*/
void AttEncoder::InitModel(Config& config)
{

    devID = config.devID;
    nlayer = config.nEncLayer;
    eSize = config.embSize;
    hSize = config.modelSize;
    vSize = config.srcVocabSize;
    preNorm = config.preNorm;
    dropoutP = config.dropout;

    CheckNTErrors(nlayer >= 1, "We have one encoding layer at least!");
    CheckNTErrors(vSize > 1, "Set vocabulary size by \"-vsize\"");

    /* embedding model */
    embedder.InitModel(config);

    selfAtt = new Attention[nlayer];
    fnns = new FNN[nlayer];
    attLayerNorms = new LN[nlayer];
    fnnLayerNorms = new LN[nlayer];

    if (preNorm)
        encoderLayerNorm = new LN;

    /* initialize the stacked layers */
    for (int i = 0; i < nlayer; i++) {
        selfAtt[i].InitModel(config);
        fnns[i].InitModel(config);
        attLayerNorms[i].InitModel(config);
        fnnLayerNorms[i].InitModel(config);
    }
    if (preNorm)
        encoderLayerNorm->InitModel(config);
}

/*
make the encoding network
>> input - the input tensor of the encoder
>> mask - the mask that indicate each position is valid
>> maskEncDec - no use
>> isTraining - indicates whether the model is used for training
<< return - the output tensor of the encoder
*/
XTensor AttEncoder::Make(XTensor& input, XTensor* mask, XTensor& maskEncDec, bool isTraining)
{
    XTensor x;

    x = embedder.Make(input, false, isTraining);

    /* dropout */
    if (isTraining && dropoutP > 0)
        x = Dropout(x, dropoutP);

    for (int i = 0; i < nlayer; i++) {
        XTensor att;
        XTensor fnn;
        XTensor res;
        XTensor attnBefore;
        XTensor attnAfter;
        XTensor fnnBefore;

        /* layer normalization with pre-norm for self-attn */
        attnBefore = LayerNorm(x, attLayerNorms[i], preNorm, true, false);

        /* self attention */
        att = selfAtt[i].Make(attnBefore, attnBefore, attnBefore, mask, isTraining, NULL, SELF_ATT);

        /* dropout */
        if (isTraining && dropoutP > 0)
            att = Dropout(att, dropoutP);

        /* residual connection */
        res = Sum(att, x);

        /* layer normalization with post-norm for self-attn */
        attnAfter = LayerNorm(res, attLayerNorms[i], preNorm, false, true);

        /* layer normalization with pre-norm for fnn */
        fnnBefore = LayerNorm(attnAfter, fnnLayerNorms[i], preNorm, true, false);

        /* fnn */
        fnn = fnns[i].Make(fnnBefore, isTraining);

        /* dropout */
        if (isTraining && dropoutP > 0)
            fnn = Dropout(fnn, dropoutP);

        /* residual connection */
        res = Sum(fnn, attnAfter);

        /* layer normalization with post-norm for fnn */
        x = LayerNorm(res, fnnLayerNorms[i], preNorm, false, true);
    }
    if (preNorm)
        return encoderLayerNorm->Make(x);

    return x;
}

/*
make the encoding network
>> input - the input tensor of the encoder
>> mask - the mask that indicate each position is valid
>> maskEncDec - no use
>> isTraining - indicates whether the model is used for training
<< return - the output tensor of the encoder
*/
XTensor AttEncoder::MakeFast(XTensor& input, XTensor* mask, XTensor& maskEncDec, bool isTraining)
{
    XTensor x;

    x = embedder.Make(input, false, isTraining);

    /* dropout */
    if (isTraining && dropoutP > 0)
        x = Dropout(x, dropoutP);

    for (int i = 0; i < nlayer; i++) {
        XTensor res;

        res = x;

        /* layer normalization with pre-norm for self-attn */
        x = attLayerNorms[i].Make(x);

        /* self attention */
        x = selfAtt[i].Make(x, x, x, mask, isTraining, NULL, SELF_ATT);

        /* dropout */
        if (isTraining && dropoutP > 0)
            x = Dropout(x, dropoutP);

        /* residual connection */
        x = Sum(res, x);

        res = x;

        /* layer normalization with pre-norm for fnn */
        x = fnnLayerNorms[i].Make(x);

        /* fnn */
        x = fnns[i].Make(x, isTraining);

        /* dropout */
        if (isTraining && dropoutP > 0)
            x = Dropout(x, dropoutP);

        /* residual connection */
        x = Sum(res, x);
    }
    x = encoderLayerNorm->Make(x);

    return x;
}

/*
make the encoding network (wrapper)
>> input - the input tensor of the encoder
>> mask - the mask that indicate each position is valid
>> isTraining - indicates whether the model is used for training
<< return - the output tensor of the encoder
*/
XTensor AttEncoder::Make(XTensor& input, XTensor* mask, bool isTraining)
{
    XTensor nothing;

    return Make(input, mask, nothing, isTraining);
}

}