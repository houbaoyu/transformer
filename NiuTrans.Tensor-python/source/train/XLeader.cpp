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
*/

#include "XLeader.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts {

/* constructor */
XLeader::XLeader()
{
    id = -1;
}

/* de-constructor */
XLeader::~XLeader()
{
}

/* intialize the leader */
void XLeader::Init()
{
    for (int i = 0; i < jworkers.count; i++)
        delete (XWorkerJob*)jworkers.GetItem(i);
    jworkers.Clear();

    for (int i = 0; i < cworkers.count; i++)
        delete (XWorkerCollect*)cworkers.GetItem(i);
    cworkers.Clear();

    for (int i = 0; i < uworkers.count; i++)
        delete (XWorkerUpdate*)uworkers.GetItem(i);
    uworkers.Clear();

    for (int i = 0; i < bworkers.count; i++)
        delete (XWorkerBroadcast*)bworkers.GetItem(i);
    bworkers.Clear();

    serverRecord.Clear();
}

/* set id */
void XLeader::SetID(int myID)
{
    id = myID;
}

/* get id */
int XLeader::GetID()
{
    return id;
}

/* 
Set the server model. It distributes the server-side parameters on different devices.
>> config - the configuration
>> model - the base model
>> memberModels - the models that run on different devices. We can place
                  the server-side parameters on different member models.
*/
void XLeader::SetServerModel(XConfig * config, XModel * model, XList * memberModels)
{
    serverModel.Clear();
    for (int i = 0; i < model->paramNum; i++) {
        XTensor * param = model->params[i].param;
        serverModel.AddParam(param);
    }

    /* TODO: we can place parameters on different devices */
}

/* 
set the server model. It distributes the server-side parameters on different devices.
>> config - the configuration
>> model - the base model*/
void XLeader::SetServerModel(XConfig * config, XModel * model)
{
    XList members;
    for (int i = 0; i < jworkers.count; i++) {
        XModel * member = ((XWorkerJob*)jworkers[i])->GetModel();
        members.Add(member);
    }

    SetServerModel(config, model, &members);
}
    
/* initialize the models for running them */
void XLeader::InitForRun()
{
    serverModel.InitForRun();

    for (int i = 0; i < jworkers.count; i++) {
        XModel* model = ((XWorkerJob*)jworkers[i])->GetModel();
        model->InitForRun();
    }

    XList workers;
    workers.AddList(&jworkers);
    workers.AddList(&cworkers);
    workers.AddList(&uworkers);
    workers.AddList(&bworkers);

    for (int i = 0; i < workers.count; i++) {
        XWorker* worker = (XWorker*)workers[i];
        CheckNTErrors(worker->IsEmpty(), "Something is wrong with the finishedQueue!");
    }
}

/*
wait for finished states (i.e., all workers finish their jobs)
>> activeJobWorkers - indicates whether each job worker is active
*/
void XLeader::WaitForFinishing(const int* activeJobWorkers)
{
    int activeCount = 0;
    for (int i = 0; i < jworkers.count; i++) {
        if (activeJobWorkers[i] > 0) {
            XWorker* worker = (XWorker*)jworkers[i];
            worker->DequeueFinishedJob();
            activeCount++;
        }
    }

    if (activeCount > 0) {
        for (int i = 0; i < cworkers.count; i++) {
            XWorker* worker = (XWorker*)cworkers[i];
            worker->DequeueFinishedJob();
        }

        for (int i = 0; i < uworkers.count; i++) {
            XWorker* worker = (XWorker*)uworkers[i];
            for (int j = 0; j < serverModel.paramNum; j++)
                worker->DequeueFinishedJob();
        }

        for (int i = 0; i < bworkers.count; i++) {
            XWorker* worker = (XWorker*)bworkers[i];
            for (int j = 0; j < serverModel.paramNum; j++)
                worker->DequeueFinishedJob();
        }
    }
}

/* get loss */
float XLeader::GetLoss()
{
    return serverRecord.lossAll;
}
    
/* get sample number */
int XLeader::GetSampleNum()
{
    return serverRecord.sampleNum;
}

/* get prediction number */
int XLeader::GetPredictNum()
{
    return serverRecord.predictNum;
}

/* 
set the communication mode 
>> myMode - the mode
*/
void XLeader::SetMode(XLEADER_MODE myMode)
{
    mode = myMode;
}

/* set the flag of instant run */
void XLeader::SetInstantRun(bool flag)
{
    for (int i = 0; i < jworkers.count; i++) {
        XWorkerJob * worker = (XWorkerJob*)jworkers.GetItem(i);
        worker->SetInstantRun(flag);
    }

    for (int i = 0; i < cworkers.count; i++) {
        XWorkerJob * worker = (XWorkerJob*)cworkers.GetItem(i);
        worker->SetInstantRun(flag);
    }

    for (int i = 0; i < uworkers.count; i++) {
        XWorkerJob * worker = (XWorkerJob*)uworkers.GetItem(i);
        worker->SetInstantRun(flag);
    }

    for (int i = 0; i < bworkers.count; i++) {
        XWorkerJob * worker = (XWorkerJob*)bworkers.GetItem(i);
        worker->SetInstantRun(flag);
    }
}

/* start the workers */
void XLeader::Start()
{
    serverModel.CheckParam();

    for (int i = 0; i < jworkers.count; i++) {
        XWorkerJob * worker = (XWorkerJob*)jworkers.GetItem(i);
        worker->GetModel()->CheckParam();
        worker->Start();
    }

    for (int i = 0; i < cworkers.count; i++) {
        XWorkerJob * worker = (XWorkerJob*)cworkers.GetItem(i);
        worker->Start();
    }

    for (int i = 0; i < uworkers.count; i++) {
        XWorkerJob * worker = (XWorkerJob*)uworkers.GetItem(i);
        worker->Start();
    }

    for (int i = 0; i < bworkers.count; i++) {
        XWorkerJob * worker = (XWorkerJob*)bworkers.GetItem(i);
        worker->Start();
    }
}

/* 
add a number of job workers (given their device ids) 
>> model - the neural network
>> n - number of the models
>> ids - the array of device ids
*/
void XLeader::AddJobWorker(XModel * model, int n, int * ids)
{
    /* we keep the input model */
    if (n >= 1) {
        XWorkerJob * worker = new XWorkerJob();
        worker->SetModel(model);
        jworkers.Add(worker);
    }

    /* we clone the input model */
    for (int i = 0; i < n - 1; i++) {
        XWorkerJob * worker = new XWorkerJob();
        worker->SetModel(model->Clone(ids[i]));
        jworkers.Add(worker);
    }
}

/* 
add a data-collecting worker 
>> mode - the data-transfer mode of the worker
*/
void XLeader::AddJobCollectWorker(DATA_COLLECT_TYPE mode)
{
    XWorkerCollect * worker = new XWorkerCollect();
    worker->SetCollectMode(mode);
    cworkers.Add(worker);
}

/* 
add a model-update worker 
>> model - the model
>> optimizer - the optimizer
*/
void XLeader::AddJobUpdateWorker(XModel * model, XOptimizer * optimizer)
{
    XWorkerUpdate * worker = new XWorkerUpdate();
    worker->SetOptimizer(optimizer);
    uworkers.Add(worker);
}

/* add a data-broadcasting worker */
void XLeader::AddJobBroadcastWorker()
{
    XWorkerBroadcast * worker = new XWorkerBroadcast();
    bworkers.Add(worker);
}

/* 
run the model (for one time). Basically this is a map-reduce process.
>> config - the configuration
>> dataDistributor - data distributor
>> model - the neural network that we want to run
>> optimizer - the optimization method
<< return - if we can fetch the new data
*/
bool XLeader::Run(XConfig * config, DataDistributeBase * dataDistributor,
                  XModel * model, XOptimizer * optimizer)
{
    CheckNTErrors(jworkers.count > 0, "No jworkers!");
    CheckNTErrors(cworkers.count > 0, "No cworkers!");
    CheckNTErrors(uworkers.count > 0, "No uworkers!");
    CheckNTErrors(bworkers.count > 0, "No bworkers!");

    bool isDataOK = true;
    int activeJobCount = 0;
    int* active = new int[jworkers.count];
    
    InitForRun();

    for (int i = 0; i < jworkers.count; i++)
        active[i] = 0;

    /* Feed the input to each worker and geneate the output.
       For each worker, we define a job queue and enqueue jobs
       into it. 
    */
    for (int i = 0; i < jworkers.count; i++) {
        XWorkerJob * worker = (XWorkerJob*)jworkers[i];
        XModel * jmodel = worker->GetModel();

        /* get a batch of samples */
        bool fetched = dataDistributor->GetBatchSimple(worker->GetInput(), worker->GetGold()); 

        if (!fetched)
            isDataOK = false;
        else {
            /* job in queue 1: refresh the model */
            worker->AddJobRefresh(jmodel);

            /* job in queue 1: run the model */
            worker->AddJobNeuralNet(jmodel, 
                                    worker->GetInput(), worker->GetOutput(), 
                                    worker->GetGold(), worker->GetLoss());

            /* job in queue 1: make a record of the run */
            worker->AddJobRecord(&serverRecord);

            /* job in queue 1: mark finished */
            worker->AddJobEnqueueFinished();

            active[i] = 1;
            activeJobCount++;
        }
    }

    if (activeJobCount > 0) {
        /* workers */
        XWorkerCollect * collecter = (XWorkerCollect*)cworkers.GetItem(0);
        XWorkerUpdate * updater = (XWorkerUpdate*)uworkers.GetItem(0);
        XWorkerBroadcast * broadcaster = (XWorkerBroadcast*)bworkers.GetItem(0);

        /* member models that are active in this run */
        XList members(jworkers.count);

        /* all member models */
        XList membersAll(jworkers.count);

        /* records of the active member models */
        XList memberRecords(jworkers.count);

        for (int i = 0; i < jworkers.count; i++) {
            XWorkerJob* worker = (XWorkerJob*)jworkers[i];
            membersAll.Add(worker->GetModel());
            if (active[i] == 1) {
                members.Add(worker->GetModel());
                memberRecords.Add(worker->GetRecord());
            }
        }

        collecter->AddJobUpdateAll(&members, &membersAll, &serverModel, 
                                   optimizer, updater, broadcaster);
        //collecter->AddJobCollectOther(&memberRecords, &serverRecord);
        collecter->AddJobEnqueueFinished();
        
        /* jobs in queue 2: collect the (gradient) data and other stuff. This
           is a reduce process. */
        //collecter->AddJobCollect(&members, &serverModel);
        //collecter->AddJobCollectOther(&memberRecords, &serverRecord);

        /* job in queue 3: update the model */
        //updater->AddJobUpdate(&serverModel, optimizer);

        /* job in queue 4: broadcast the lastest parameters to workers. NOTE that
           we would update a worker to the laster model parameters, even if it is
           not involved in this run. */
        //broadcaster->AddJobBroadcast(&serverModel, &membersAll);

        //WaitForFinishing();
    }

    WaitForFinishing(active);

    for (int i = 0; i < jworkers.count; i++) {
        XWorkerJob * worker = (XWorkerJob*)jworkers[i];
        worker->Clear();
    }

    delete[] active;

    return isDataOK;
}

/* wait until all workers finish their job */
void XLeader::WaitForFinishing(int sleepTime)
{
    while (1) {
        bool finished = true;

        if (finished) {
            for (int i = 0; i < jworkers.count; i++) {
                XWorkerJob* worker = (XWorkerJob*)jworkers[i];
                if (worker->GetJobNum() > 0) {
                    finished = false;
                    break;
                }
            }
        }

        if (finished) {
            for (int i = 0; i < cworkers.count; i++) {
                XWorkerJob* worker = (XWorkerJob*)cworkers[i];
                if (worker->GetJobNum() > 0) {
                    finished = false;
                    break;
                }
            }
        }

        if (finished) {
            for (int i = 0; i < uworkers.count; i++) {
                XWorkerJob* worker = (XWorkerJob*)uworkers[i];
                if (worker->GetJobNum() > 0) {
                    finished = false;
                    break;
                }
            }
        }

        if (finished) {
            for (int i = 0; i < bworkers.count; i++) {
                XWorkerJob* worker = (XWorkerJob*)bworkers[i];
                if (worker->GetJobNum() > 0) {
                    finished = false;
                    break;
                }
            }
        }

        if (finished)
            break;

        XSleep(sleepTime);
    }
}

} /* end of the nts (NiuTrans.Tensor) namespace */
