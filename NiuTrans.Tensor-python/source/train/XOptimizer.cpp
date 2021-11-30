/*
* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2016-2021
* Natural Language Processing Lab, Northeastern University
* and
* NiuTrans Research
* All rights reserved.
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
* This class define the template of the update rule in gradient based methods
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-01
*/

#include "XOptimizer.h"
#include "../tensor/core/CHeader.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* constructor */
XOptimizer::XOptimizer()
{
    Clear();
}

/* de-constructor */
XOptimizer::~XOptimizer()
{
}

/* 
initialize the optimizer 
>> config - the configuration
*/
void XOptimizer::Init(XConfig &config)
{
    nstep = config.GetInt("nstep", 100000);
    nepoch = config.GetInt("nepoch", 50);
    lrate = config.GetFloat("lrate", 0.1F);
}

/* clear the optimizer */
void XOptimizer::Clear()
{
    nstep = 0;
    nepoch = 0;
    lrate = 0;
}

void XOptimizer::ShowSettings()
{
    XPRINT(1, stderr, "[INFO] Optimizer Setup:\n");
    XPRINT1(1, stderr, "       nstep = %d\n", nstep);
    XPRINT1(1, stderr, "       nepoch = %d\n", nepoch);
    XPRINT1(1, stderr, "       lrate = %.3f\n", lrate);
}

/* 
prepare for the update 
>> model - the model that we want to update
*/
void XOptimizer::Prepare(XModel * model)
{
}

/* 
record the update 
>> model - the model that we want to update
*/
void XOptimizer::Note(XModel * model)
{
    nstep++;
}

/* 
update a parameter matrix
>> param - the parameter matrix
>> gard - the gradient
>> pid - the id of the parameter matrix
*/
void XOptimizer::UpdateParam(XTensor * param, XTensor * grad, int pid)
{
    /* the delta rule
       \theta_new = \theta_old - \grad * \lrate */
    _Sum(param, grad, param, -lrate);
}

}
