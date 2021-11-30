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

#include "Embedding.h"
#include "LayerNorm.h"
#include "../Utility.h"
#include "../../../tensor/core/CHeader.h"

namespace nmt
{

/* constructor */
LN::LN()
{
    devID = -1;
    d = 0;
}

/* de-constructor */
LN::~LN()
{
}

/*
initialize the model
>> argc - number of arguments
>> argv - list of pointers to the arguments
>> config - configurations of the model
*/
void LN::InitModel(Config& config)
{
    devID = config.devID;

    d = config.modelSize;

    InitTensor1D(&w, d, X_FLOAT, devID);
    InitTensor1D(&b, d, X_FLOAT, devID);
    w.SetDataRand(1.0F, 1.0F);
    b.SetZeroAll();

    w.SetDataFixed(1);
}

/*
make the network
>> input - the input tensor
>> return - layer normalization output
*/
XTensor LN::Make(XTensor& input)
{
    XTensor& x = input;
    XTensor xn;
    XTensor mean;
    XTensor variance;
    XTensor standard;
    XTensor meanFilled;
    XTensor standardFilled;

    TENSOR_DATA_TYPE dataType = input.dataType;

    if (dataType == X_FLOAT16) {
        /* reduce functions can only run with FP32 */
        x = ConvertDataType(input, X_FLOAT);
    }

    /* \mu = (sum_i x_i)/m */
    mean = ReduceMean(x, x.order - 1);

    /* \sigma = (sum_i (x_i - \mu)^2)/m */
    variance = ReduceVariance(x, x.order - 1, mean);

    /* standard = sqrt(variance) */
    standard = Power(variance, 0.5F);

    /* unsqueeze mean and standard deviation to fit them into
       the same shape of x */
    meanFilled = Unsqueeze(mean, x.order - 1, x.GetDim(-1));
    standardFilled = Unsqueeze(standard, x.order - 1, x.GetDim(-1));

    /* x' = (x - \mu)/standard */
    xn = (x - meanFilled) / standardFilled;

    if (dataType != mean.dataType) {
        x = ConvertDataType(x, dataType);
        xn = ConvertDataType(xn, dataType);
    }

    /* result = x' * w + b   */
    return xn * w + b;
}

}