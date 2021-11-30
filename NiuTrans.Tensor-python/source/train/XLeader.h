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
* A "leader" manages a number of "workers". The leader recieves jobs from
* the central server (can be remote), or acts as an independent server itself. 
* For workers, the leader is the one who issues orders and organizes them. 
* Note that the leader and workers must be on the same machine. In case of 
* multi-machine training, one can deploy different leaders on different 
* machines. BUT, at this time, we need an additional way of distributing 
* data across machines.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-02-25
* We will go for a business trip. The first trip after the Spring Festival.
*/

#ifndef __XLEADER_H__
#define __XLEADER_H__

#include "XModel.h"
#include "XOptimizer.h"
#include "XBaseTemplate.h"
#include "XWorkerJob.h"
#include "XWorkerCollect.h"
#include "XWorkerUpdate.h"
#include "XWorkerBroadcast.h"
#include "../tensor/XConfig.h"
#include "../tensor/XList.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#define MAX_NUM_OF_WORKERS 1024
#define SLEEP_TIME_IN_WAITING_FOR_JOBS 20

/* 
conmmunication mode of a leader. This offers a way of organizing a hierachy of the work
1) run as a standalone program
2) give orders to another leader (probably remote)
3) recieve orders from anothe leader (probably remote)
4) give (and recieve) orders to (and from) different leaders
*/
enum XLEADER_MODE { XLEADER_STANDALONE, XLEADER_SEND, XLEADER_RECIEVE, XLEADER_SEND_AND_RECIEVE };

/* a leader who manages workers */
class XLeader
{
protected:
    /* id of the leader */
    int id;

    /* a model that keeps the parameters (as a server) */
    XModel serverModel;

    /* a record that keeps the information of the run */
    XNNRecord serverRecord;

    /* communication mode */
    XLEADER_MODE mode;

    /* job workers */
    XList jworkers;

    /* data-collecting workers */
    XList cworkers;

    /* model-update workers */
    XList uworkers;

    /* data-broadcasting workers */
    XList bworkers;

public:
    /* constructor */
    XLeader();

    /* de-constructor */
    ~XLeader();

    /* intialize the leader */
    void Init();

    /* set id */
    void SetID(int myID);

    /* get id */
    int GetID();

    /* set the server model */
    void SetServerModel(XConfig * config, XModel * model, XList * memberModels);

    /* set the server model */
    void SetServerModel(XConfig * config, XModel * model);
    
    /* initialize the models for running them */
    void InitForRun();

    /* wait for finished states (i.e., all workers finish their jobs) */
    void WaitForFinishing(const int * activeJobWorkers);

    /* get loss */
    float GetLoss();
    
    /* get sample number */
    int GetSampleNum();

    /* get prediction number */
    int GetPredictNum();

    /* start the workers */
    void Start();

    /* set the communication mode */
    void SetMode(XLEADER_MODE myMode);

    /* set the flag of instant run */
    void SetInstantRun(bool flag = true);
    
    /* add a number of job workers (given their device ids) */
    void AddJobWorker(XModel * model, int n, int * ids);

    /* add a data-collecting worker */
    void AddJobCollectWorker(DATA_COLLECT_TYPE mode = DATA_COLLECT_P2P);

    /* add a model-update worker */
    void AddJobUpdateWorker(XModel * model, XOptimizer * optimizer);

    /* add a data-broadcasting worker */
    void AddJobBroadcastWorker();

    /* run the model (for one time) */
    bool Run(XConfig * config, DataDistributeBase * dataDistributor, 
             XModel * model, XOptimizer * optimizer);

    /* wait until all workers finish their job */
    void WaitForFinishing(int sleepTime = SLEEP_TIME_IN_WAITING_FOR_JOBS);
};

}

#endif // __XLEADER_H__
