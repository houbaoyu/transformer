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
* The worker that collects data from workers.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-02
* minus 10 degrees centigrade comes again!
*/

#ifndef __XWORKERCOLLECT_H__
#define __XWORKERCOLLECT_H__

#include "XWorker.h"
#include "XModel.h"
#include "XWorkerJob.h"
#include "XWorkerUpdate.h"
#include "XWorkerBroadcast.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#define SLEEP_TIME_IN_COLLECTING 5
#define SLEEP_TIME_IN_COLLECTING_OTHER 5

/*
data collection method
1) point-to-point
2) reduce sum
3) all-reduce
*/
enum DATA_COLLECT_TYPE { DATA_COLLECT_P2P, DATA_COLLECT_REDUCESUM};

/* The class defines the collecting-data worker. It collect (gradient) data
   from workers for the leader (server). */
class XWorkerCollect : public XWorker
{
protected:
    DATA_COLLECT_TYPE collectMode;

public:
    /* constructor */
    XWorkerCollect();

    /* de-constructor */
    ~XWorkerCollect();

    /* set the collection type */
    void SetCollectMode(DATA_COLLECT_TYPE myMode);

    /* collect the gradient data, update the parameters, and broadcast the 
       new parameters to all models. NOTE that this method just collects graidents
       from member models. Then it calls an XWorkerUpdate to update the parameters.
       The XWorkerUpdate also calls an XWorkerBroadcast to broadcast the new parameter
       to member models back. */
    void UpdateDataAll(XList * memberActive, XList * memberAll, XModel * server, 
                       XOptimizer * optimizer, XWorkerUpdate * updater, XWorkerBroadcast * broadcaster, 
                       int sleepTime);

    /* wrapper of UpdateDataAll */
    static
    void UpdateAll(XList * args);

    /* P2P data collection */
    void CollectP2P(XTensor * source, XTensor * target);

    /* sum-reduce for given tensors */
    void CollectReduceSum(XList * source, XTensor * target);

    /* all-reduce */
    void CollectAllReduce(XList * all);

    /* add a new job of collecting data, update the parameter and broadcast the new parameter */
    bool AddJobUpdateAll(XList * memberActive, XList * memberAll, XModel * server,
                         XOptimizer * optimizer, XWorkerUpdate * updater, XWorkerBroadcast * broadcaster);

    /* add a new job of collecting data */
    bool AddJobCollect(XList * sourceList, XModel * target);

    /* collect the data of the run (i.e., loss). This is a reducer. */
    void CollectOtherData(XList * sourceList, XNNRecord * target, int sleepTime);

    /* wrapper of CollectOtherData */
    static
    void CollectOther(XList * args);

    /* add a new job of collecting data of the run (i.e., loss) */
    bool AddJobCollectOther(XList * sourceList, XNNRecord * target);
};

}

#endif
