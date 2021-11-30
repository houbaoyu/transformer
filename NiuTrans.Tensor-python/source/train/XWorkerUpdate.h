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
* The worker that updates the model.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-01
*/

#ifndef __XWORKERUPDATE_H__
#define __XWORKERUPDATE_H__

#include "XWorker.h"
#include "XOptimizer.h"
#include "XWorkerBroadcast.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#define SLEEP_TIME_IN_MODEL_UPDATE 5

/* The class defines the model-update worker */
class XWorkerUpdate : public XWorker
{
protected:
    /* the optimizer */
    XOptimizer * optimizer;

public:
    /* constructor */
    XWorkerUpdate();

    /* de-constructor */
    ~XWorkerUpdate();

    /* set the optimizer */
    void SetOptimizer(XOptimizer * myOptimizer);

    /* get the optimizer */
    XOptimizer * GetOptimizer();

    /* update the parameter */
    void UpdateParameter(XModel * server, XList * members, int pid,
                         XOptimizer * optimizer, XWorkerBroadcast * broadcaster);

    /* update the model */
    void UpdateModel(XModel * model, XOptimizer * optimizer, int sleepTime);

    /* wrapper of UpdateParameter */
    static
    void UpdateSingle(XList * args);

    /* wrapper of UpdateModel */
    static
    void Update(XList * args);

    /* add a new job of model update (for a parameter) */
    bool AddJobUpdateSingle(XModel * model, XList * members, int pid,
                            XOptimizer * optimizer, XWorkerBroadcast * broadcaster);

    /* add a new job of model update */
    bool AddJobUpdate(XModel * model, XOptimizer * optimizer);
};

}

#endif
