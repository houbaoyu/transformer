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
* The worker that boradcast the lastest parameters from the server to
* the workers.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-03
*/


#include "XWorkerBroadcast.h"
#include "../tensor/core/CHeader.h"

namespace nts { // namespace nts(NiuTrans.Tensor)


/* constructor */
XWorkerBroadcast::XWorkerBroadcast()
{
}

/* de-constructor */
XWorkerBroadcast::~XWorkerBroadcast()
{
}

/* set the broadcasting type */
void XWorkerBroadcast::SetBroadcastMode(DATA_BROADCAST_TYPE myMode)
{
    broadcastMode = myMode;
}

/* 
broadcast data for a parameter 
>> source - the data (as a model) that we want to broadcast
>> targetList - the target places that we recieve the data
>> pid - the parameter index
*/
void XWorkerBroadcast::BroadcastDataSingle(XModel * source, XList * targetList, int pid)
{
    CheckNTErrors(source->params[pid].flag == PARAM_STATE_UPDATED,
                  "The parameter is not ready for broadcasting");

    for (int i = 0; i < targetList->count; i++) {
        XModel * target = (XModel*)targetList->GetItem(i);

        /* data transmit */
        BroadcastP2P(source->params[pid].param, target->params[pid].param);

        /* update the flag */
        target->params[pid].flag = PARAM_STATE_UPDATED;
    }
}

/* 
broadcast data for a model
>> source - the data that we want to broadcast
>> targetList - the target places that we recieve the data
>> sleepTime - the waiting time in broadcasting
*/
void XWorkerBroadcast::BroadcastData(XModel * source, XList * targetList, int sleepTime)
{
    int finished = 0;
    int * finishedFlag = new int[source->paramNum];
    memset(finishedFlag, 0, sizeof(int) * source->paramNum);

    /* check */
    for (int i = 0; i < targetList->count; i++) {
        XModel * target = (XModel*)targetList->GetItem(i);
        CheckNTErrors(source->paramNum == target->paramNum, "Incompatiable models!");
    }

    /* the major body of broadcasting */
    while (1) {
        for (int i = 0; i < source->paramNum; i++) {
            if (source->params[i].flag == PARAM_STATE_UPDATED && finishedFlag[i] == 0) {

                /* broadcasting */
                BroadcastDataSingle(source, targetList, i);

                /* counting */
                finished += targetList->count;
                finishedFlag[i] = 1;
            }
        }

        if (finished == source->paramNum * targetList->count)
            break;

        XSleep(sleepTime);
    }

    delete[] finishedFlag;
}

/* 
wrapper of BroadcastDataSingle 
>> args - the list of arguments
*/
void XWorkerBroadcast::BroadcastSingle(XList * args)
{
    XWorkerBroadcast * broadcaster = (XWorkerBroadcast*)args->GetItem(0);
    XModel * source = (XModel*)args->GetItem(1);

    /* target models */
    int targetNum = args->GetItemInt(2);
    XList target;
    for (int i = 0; i < targetNum; i++) {
        XModel * model = (XModel*)args->GetItem(3 + i);
        target.Add(model);
    }

    /* parameter index */
    int p = args->GetInt(3 + targetNum);

    broadcaster->BroadcastDataSingle(source, &target, p);
}

/* 
wrapper of BroadcastData 
>> args - the list of arguments
*/
void XWorkerBroadcast::Broadcast(XList * args)
{
    //fprintf(stderr, "broadcast 0\n");
    XWorkerBroadcast * broadcaster = (XWorkerBroadcast*)args->GetItem(0);
    XModel * source = (XModel*)args->GetItem(1);

    /* target models */
    int targetNum = args->GetItemInt(2);
    XList target;
    for (int i = 0; i < targetNum; i++) {
        XModel * model = (XModel*)args->GetItem(3 + i);
        target.Add(model);
    }

    broadcaster->BroadcastData(source, &target, SLEEP_TIME_IN_BROADCASTING);
    //fprintf(stderr, "broadcast 1\n");
}

/* 
P2P data broadcasting 
>> source - the source data
>> target - the target data
*/
void XWorkerBroadcast::BroadcastP2P(XTensor * source, XTensor * target)
{
    CheckNTErrors(source != NULL, "The source tensor should not be NULL!");
    CheckNTErrors(target != NULL, "The target tensor should not be NULL!");
    CheckNTErrors(IsSameShaped(*source, *target), "The two tensors should be of the same shape!");

    if(source != target)
        CopyValues(*source, *target);
}

/* 
add a new job of broadcasting data (for a parameter)
>> source - the data that we want to broadcast
>> targetList - the target places that we recieve the data
>> pid - the parameter index
*/
bool XWorkerBroadcast::AddJobBroadcastSingle(XModel * source, XList * targetList, int pid)
{
    CheckNTErrors(source != NULL, "no input source tensor!");
    CheckNTErrors(targetList != NULL, "no input target tensor list!");
    CheckNTErrors(pid >= 0 && pid < source->paramNum, "illegal parameter index!");

    XList args;
    args.Add(this);
    args.Add(source);
    args.AddInt(targetList->count);
    args.AddList(targetList);
    args.AddInt(pid);

    if (isInstantRun)
        XWorkerBroadcast::BroadcastSingle(&args);
    else
        queue.EnqueueJob((void*)(char*)XWorkerBroadcast::BroadcastSingle, &args);

    return true;
}

/* 
add a new job of broadcasting data (for a model)
>> source - the data that we want to broadcast
>> targetList - the target places that we recieve the data
*/
bool XWorkerBroadcast::AddJobBroadcast(XModel * source, XList * targetList)
{
    CheckNTErrors(source != NULL, "no input source tensor!");
    CheckNTErrors(targetList != NULL, "no input target tensor list!");

    XList args;
    args.Add(this);
    args.Add(source);
    args.AddInt(targetList->count);
    args.AddList(targetList);

    if (isInstantRun)
        XWorkerBroadcast::Broadcast(&args);
    else
        queue.EnqueueJob((void*)(char*)XWorkerBroadcast::Broadcast, &args);

    return true;
}

}
