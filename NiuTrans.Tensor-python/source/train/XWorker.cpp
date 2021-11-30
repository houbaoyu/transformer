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
* The base class of worker. It maintains a job queue and offers utilities
* of controlling the working pipeline.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-02-24
*/

#include "XWorker.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts {

/* constructor */
XWorker::XWorker()
{
    devID = -1;
    id = -1;
    state = XWORKER_UNSTARTED;
    isInstantRun = false;
}

/* de-constructor */
XWorker::~XWorker()
{
    Stop();
}

/* set device id */
void XWorker::SetDeviceID(int myDevID)
{
    devID = myDevID;
}

/* get device id */
int XWorker::GetDeviceID()
{
    return devID;
}

/* set worker id */
void XWorker::SetID(int myID)
{
    id = myID;
}

/* get worker id */
int XWorker::GetID()
{
    return id;
}

/* set the flag of instant run */
void XWorker::SetInstantRun(bool flag)
{
    isInstantRun = flag;
}

/* 
enqueue a new job 
>> job - the job function
>> jobArgs - the arguments of the function
*/
void XWorker::AddJob(void * job, XList * jobArgs)
{
    queue.EnqueueJob(job, jobArgs);
}

/* start the work */
void XWorker::Start()
{
    queue.RunJobConsumer();
}

/* stop the work */
void XWorker::Stop()
{
    queue.StopJobConsumer();
}

/* get the number of remaining jobs */
int XWorker::GetJobNum()
{
    return queue.GetJobNum();
}

/* whether the job queue is empty? */
bool XWorker::IsEmpty()
{
    return queue.IsEmpty();
}

/* enqueue a counting job of a finished job */
void XWorker::EnqueueFinishedJob()
{
    finishedQueue.Enqueue(NULL);
}

/* dequeue a counting job of a finished job */
void XWorker::DequeueFinishedJob()
{
    finishedQueue.Dequeue();
}

/* wrapper of EnqueueFinished() */
void XWorker::EnqueueFinished(XList* args)
{
    XWorker* worker = (XWorker*)args->GetItem(0);
    worker->EnqueueFinishedJob();
}

/* wrapper of DequeueFinished() */
void XWorker::DequeueFinished(XList* args)
{
    XWorker* worker = (XWorker*)args->GetItem(0);
    worker->DequeueFinishedJob();
}

/* add a job of enqueuing a counting a finished job */
void XWorker::AddJobEnqueueFinished()
{
    XList args;
    args.Add(this);

    if (isInstantRun)
        XWorker::EnqueueFinished(&args);
    else
        queue.EnqueueJob((void*)(char*)XWorker::EnqueueFinished, &args);
}

/* add a job of dequeuing a counting a finished job */
void XWorker::AddJobDequeueFinished()
{
    XList args;
    args.Add(this);

    if (isInstantRun)
        XWorker::DequeueFinished(&args);
    else
        queue.EnqueueJob((void*)(char*)XWorker::DequeueFinished, &args);

}

} /* end of the nts (NiuTrans.Tensor) namespace */
