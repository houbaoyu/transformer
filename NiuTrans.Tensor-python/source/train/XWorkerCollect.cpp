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
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-01
*/

#include "XWorkerCollect.h"
#include "../tensor/core/CHeader.h"

namespace nts { // namespace nts(NiuTrans.Tensor)


/* constructor */
XWorkerCollect::XWorkerCollect()
{
    collectMode = DATA_COLLECT_P2P;
}

/* de-constructor */
XWorkerCollect::~XWorkerCollect()
{
}

/* set the collection type */
void XWorkerCollect::SetCollectMode(DATA_COLLECT_TYPE myMode)
{
    collectMode = myMode;
}

/* 
collect the gradient data, update the parameters, and broadcast the
new parameters to all models. NOTE that this method just collect graident
from member models. Then it calls an XWorkerUpdate to update the parameters.
The XWorkerUpdate also calls an XWorkerBroadcast to broadcast the new parameter
to member models back. 
>> memberActive - member models that are active, i.e., have generated gradients
>> memberAll -  all member models
>> server - the server model
>> optimizer - the optimizer
>> updater - the worker that updates the parameters
>> broadcaster - the worker that broadcasts the new parameters to all member
                 models
>> sleepTime - waiting time in collecting
*/
void XWorkerCollect::UpdateDataAll(XList * memberActive, XList * memberAll, XModel * server,
                                   XOptimizer * optimizer, XWorkerUpdate * updater, 
                                   XWorkerBroadcast * broadcaster, int sleepTime)
{
    int finished = 0;

    for (int j = 0; j < server->paramNum; j++)
        server->params[j].flag = PARAM_STATE_NOT_READY;

    /* check */
    for (int i = 0; i < memberAll->count; i++) {
        XModel * source = (XModel*)memberAll->GetItem(i);
        CheckNTErrors(source->paramNum == server->paramNum, "Incompatiable models!");
    }

    for (int i = 0; i < memberActive->count; i++) {
        XModel * source = (XModel*)memberActive->GetItem(i);
        CheckNTErrors(source->paramNum == server->paramNum, "Incompatiable models!");
    }

    /* counts how many member models are collect for each parameters */
    int * finishedCount = new int[server->paramNum];
    memset(finishedCount, 0, sizeof(int) * server->paramNum);

    /* This is a simple implementation of the wait-and-collect process. But
       there is a risk that some models are not available, that is, the
       loop would never stop. A solution might be that we force the loop
       to break after waiting for a short time. */
    while (1) {
        if (collectMode == DATA_COLLECT_P2P) {
            for (int j = 0; j < server->paramNum; j++) {

                XParamKeeper &paramServer = server->params[j];

                /* tp[j]->isGradFinished is true only if the model finishes the computation
                (in another process) */
                if (paramServer.flag != PARAM_STATE_NOT_READY || !paramServer.param->isGradFinished)
                    continue;

                /* check if all the models (or part of them) are ready */
                for (int i = 0; i < memberActive->count; i++) {
                    XModel * source = (XModel*)memberActive->GetItem(i);
                    XParamKeeper &paramSource = source->params[j];

                    /* sp[j]->isGradFinished is true only if the model finishes the computation
                    (in another process) */
                    if (paramSource.flag == PARAM_STATE_NOT_READY && paramSource.param->isGradFinished) {

                        /* data transmit */
                        CollectP2P(paramSource.param->grad, paramServer.param->grad);

                        /* reset the flag */
                        paramSource.flag = PARAM_STATE_COLLECTED;
                        finished++;
                        finishedCount[j]++;

                        /* we call model update (in another thread) and then
                           broadcast the new parameters to member models 
                           (in another thread) */
                        if (finishedCount[j] == memberActive->count) {
                            paramServer.flag = PARAM_STATE_COLLECTED;
                            if (updater != NULL) {
                                updater->AddJobUpdateSingle(server, memberAll, j, optimizer, broadcaster);
                                updater->AddJobEnqueueFinished();

                            }
                        }
                        else if (finishedCount[j] > memberActive->count) {
                            ShowNTErrors("Something is wrong with finishedCount!");
                        }
                    }
                }
            }
        }
        else {
            ShowNTErrors("Unsupported data collection mode!");
        }

        /* the collection finishes if all data tensors are processed */
        if (finished == server->paramNum * memberActive->count)
            break;

        XSleep(sleepTime);
    }

    delete[] finishedCount;
}

/* wrapper of UpdateDataAll */
void XWorkerCollect::UpdateAll(XList * args)
{
    XWorkerCollect * collecter = (XWorkerCollect*)args->GetItem(0);
    int activeNum = args->GetInt(1);
    
    XList memberActive;
    for (int i = 0; i < activeNum; i++) {
        XModel * member = (XModel*)args->GetItem(2 + i);
        memberActive.Add(member);
    }

    int allNum = args->GetInt(2 + activeNum);

    XList memberAll;
    for (int i = 0; i < allNum; i++) {
        XModel * member = (XModel*)args->GetItem(2 + activeNum + 1 + i);
        memberAll.Add(member);
    }

    XModel * server = (XModel*)args->GetItem(2 + activeNum + 1 + allNum);
    XOptimizer * optimizer = (XOptimizer*)args->GetItem(2 + activeNum + 1 + allNum + 1);
    XWorkerUpdate * updater = (XWorkerUpdate*)args->GetItem(2 + activeNum + 1 + allNum + 2);
    XWorkerBroadcast * broadcaster = (XWorkerBroadcast*)args->GetItem(2 + activeNum + 1 + allNum + 3);

    collecter->UpdateDataAll(&memberActive, &memberAll, server, 
                             optimizer, updater, broadcaster, 
                             SLEEP_TIME_IN_COLLECTING);
}

/* 
P2P data collection
target += source

>> source - the source tensor
>> target - the target tensor
*/
void XWorkerCollect::CollectP2P(XTensor * source, XTensor * target)
{
    CheckNTErrors(source != NULL, "The source tensor should not be NULL!");
    CheckNTErrors(target != NULL, "The target tensor should not be NULL!");
    CheckNTErrors(IsSameShaped(*source, *target), "The two tensors should be of the same shape!");

    /* target += source */
    if(source != target)
        Sum(*source, *target, *source);
}

/* 
sum-reduce for given tensors 
target += source_0
target += source_1
...
target += source_n

>> source - the source tensor
>> target - the target tensor
*/
void XWorkerCollect::CollectReduceSum(XList * source, XTensor * target)
{
    for (int i = 0; i < source->count; i++) {
        XTensor * s = (XTensor*)source->GetItem(i);
        CollectP2P(s, target);
    }
}

/* 
all-reduce: the well-known all-reduce method
every tensor is involved in every data transmition. The final outcome
is that all input tensors share the same value (i.e., the sum of them).

>> all - the tensors for sum
*/
void XWorkerCollect::CollectAllReduce(XList * all)
{
    ShowNTErrors("TODO!");
}

/* 
add a new job of collecting data, update the parameter and 
broadcast the new parameter
>> memberActive - member models that are active, i.e., have generated gradients
>> memberAll -  all member models
>> server - the server model
>> optimizer - the optimizer
>> updater - the worker that updates the parameters
>> broadcaster - the worker that broadcasts the new parameters to all member
                 models
<< return - successful or not
*/
bool XWorkerCollect::AddJobUpdateAll(XList * memberActive, XList * memberAll, XModel * server,
                                     XOptimizer * optimizer, XWorkerUpdate * updater, XWorkerBroadcast * broadcaster)
{
    CheckNTErrors(memberActive != NULL, "No input (active) member list!");
    CheckNTErrors(memberAll != NULL, "No input (all) member list!");
    CheckNTErrors(server != NULL, "No input server model!");
    CheckNTErrors(optimizer != NULL, "No input optimizer!");
    CheckNTErrors(updater != NULL, "No input updater!");
    CheckNTErrors(broadcaster != NULL, "No input broadcaster!");

    XList args;
    args.Add(this);
    args.AddInt(memberActive->count);
    args.AddList(memberActive);
    args.AddInt(memberAll->count);
    args.AddList(memberAll);
    args.Add(server);
    args.Add(optimizer);
    args.Add(updater);
    args.Add(broadcaster);

    if (isInstantRun)
        XWorkerCollect::UpdateAll(&args);
    else
        queue.EnqueueJob((void*)(char*)XWorkerCollect::UpdateAll, &args);

    return true;
}

/* 
add a new job of collecting data
>> sourceList - the list of models that we want collect data from
>> target - the destination of the collection
<< return - successful or not
*/
bool XWorkerCollect::AddJobCollect(XList * sourceList, XModel * target)
{
    CheckNTErrors(sourceList != NULL, "no input source model list!");
    CheckNTErrors(target != NULL, "no input target model!");

    XList args;
    args.Add(this);
    args.AddInt(sourceList->count);
    args.AddList(sourceList);
    args.AddInt(0);
    args.Add(target);
    args.Add(NULL);
    args.Add(NULL);
    args.Add(NULL);

    if (isInstantRun)
        XWorkerCollect::UpdateAll(&args);
    else
        queue.EnqueueJob((void*)(char*)XWorkerCollect::UpdateAll, &args);

    return true;
}

/* 
collect the data of the run (i.e., loss). This is a reducer. 
>> sourceList - the list of record
>> target - the record that we keep the reduce result
>> sleepTime - waiting time in collecting data
*/
void XWorkerCollect::CollectOtherData(XList* sourceList, XNNRecord* target, int sleepTime)
{
    int finished = 0;
    int* flags = new int[sourceList->count];
    
    for (int i = 0; i < sourceList->count; i++)
        flags[i] = 0;

    while (1) {
        for (int i = 0; i < sourceList->count; i++) {
            if (flags[i] != 0)
                continue;

            XNNRecord* source = (XNNRecord*)sourceList->GetItem(i);
            if (source->state == XWORKER_FINISHED) {
                if(target != source)
                    target->Update(*source);
                flags[i] = 1;
                finished++;
            }
        }

        if (finished == sourceList->count)
            break;

        XSleep(sleepTime);
    }

    delete[] flags;
}

/* wrapper of CollectOtherData */
void XWorkerCollect::CollectOther(XList* args)
{
    //fprintf(stderr, "collect data other 0\n");

    XWorkerCollect* collecter = (XWorkerCollect*)args->GetItem(0);
    int sourceNum = args->GetItemInt(1);

    /* the source records */
    XList source;
    for (int i = 0; i < sourceNum; i++) {
        XNNRecord * record = (XNNRecord*)args->GetItem(2 + i);
        source.Add(record);
    }

    /* the target record */
    XNNRecord* target = (XNNRecord*)args->GetItem(2 + sourceNum);

    collecter->CollectOtherData(&source, target, SLEEP_TIME_IN_COLLECTING_OTHER);

    //fprintf(stderr, "collect data other 1\n");
}

/* 
add a new job of collecting data of the run (i.e., loss) 
collect the data of the run (i.e., loss). This is a reducer.
>> sourceList - the list of record
>> target - the record that we keep the reduce result
*/
bool XWorkerCollect::AddJobCollectOther(XList* sourceList, XNNRecord* target)
{
    CheckNTErrors(sourceList != NULL, "no input source record list!");
    CheckNTErrors(target != NULL, "no input target record!");

    XList args;
    args.Add(this);
    args.AddInt(sourceList->count);
    args.AddList(sourceList);
    args.Add(target);

    if (isInstantRun)
        XWorkerCollect::CollectOther(&args);
    else
        queue.EnqueueJob((void*)(char*)XWorkerCollect::CollectOther, &args);

    return true;
}

}
