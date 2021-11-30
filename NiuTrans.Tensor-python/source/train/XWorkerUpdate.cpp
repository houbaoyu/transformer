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

#include "XWorkerUpdate.h"

namespace nts { // namespace nts (NiuTrans.Tensor)

/* constructor */
XWorkerUpdate::XWorkerUpdate()
{
    optimizer = NULL;
}

/* de-constructor */
XWorkerUpdate::~XWorkerUpdate()
{
}

/* set the optimizer */
void XWorkerUpdate::SetOptimizer(XOptimizer * myOptimizer)
{
    optimizer = myOptimizer;
}

/* get the optimizer */
XOptimizer * XWorkerUpdate::GetOptimizer()
{
    return optimizer;
}

/* 
update a parameter of a model 
>> model - the model that we want to update (on the server side)
>> members - models that would share the updated parameters
>> pid - the parameter index
>> optimizer - the optimizer
>> broadcaster - the worker that would broadcast the new parameter to members
*/
void XWorkerUpdate::UpdateParameter(XModel * server, XList * members, int pid,
                                    XOptimizer * optimizer, XWorkerBroadcast * broadcaster)
{

    CheckNTErrors(server->params[pid].flag == PARAM_STATE_COLLECTED, "The state of the parameter is wrong!");

    XTensor * param = server->params[pid].param;
    XTensor * grad = param->grad;

    CheckNTErrors(grad != NULL, "No gradient!");

    /* update the parameter */
    optimizer->UpdateParam(param, grad, pid);

    /* set the flag */
    server->params[pid].flag = PARAM_STATE_UPDATED;

    /* broadcast the new parameter to other models (in anotehr worker/thread) */
    broadcaster->AddJobBroadcastSingle(server, members, pid);
    broadcaster->AddJobEnqueueFinished();
}

/* 
update the model 
>> model - the model that we want to update
>> optimizer - the optimizer
>> sleepTime - waiting time in each update
*/
void XWorkerUpdate::UpdateModel(XModel * model, XOptimizer * optimizer, int sleepTime)
{
    int finished = 0;

    optimizer->Prepare(model);

    while (1) {
        for (int i = 0; i < model->paramNum; i++) {
            if (model->params[i].flag == PARAM_STATE_COLLECTED) {
                XTensor * param = model->params[i].param;
                XTensor * grad = param->grad;

                CheckNTErrors(grad != NULL, "No gradient!");

                /* update the parameter */
                optimizer->UpdateParam(param, grad, i);

                /* set the flag */
                model->params[i].flag = PARAM_STATE_UPDATED;
                finished++;
            }
        }

        if (finished == model->paramNum)
            break;

        XSleep(sleepTime);
    }

    optimizer->Note(model);
}

/* 
wrapper of UpdateParameter 
>> args - arguments of the update
*/
void XWorkerUpdate::UpdateSingle(XList * args)
{
    CheckNTErrors(args != NULL && args->count >= 6, "Illegal argument list!");

    XWorkerUpdate * updater = (XWorkerUpdate*)args->GetItem(0);
    XModel * server = (XModel*)args->GetItem(1);
    int memNum = args->GetInt(2);

    XList members;
    for (int i = 0; i < memNum; i++) {
        XModel * member = (XModel*)args->GetItem(3 + i);
        members.Add(member);
    }

    int pid = args->GetInt(3 + memNum);
    XOptimizer * optimizer = (XOptimizer*)args->GetItem(3 + memNum + 1);
    XWorkerBroadcast * broadcaster = (XWorkerBroadcast*)args->GetItem(3 + memNum + 2);

    updater->UpdateParameter(server, &members, pid, optimizer, broadcaster);
}

/* 
wrapper of UpdateModel
>> args - arguments of the update
*/
void XWorkerUpdate::Update(XList * args)
{
    //fprintf(stderr, "update 0\n");

    CheckNTErrors(args != NULL && args->count >= 3, "Illegal argument list!");

    XWorkerUpdate * updater = (XWorkerUpdate*)args->GetItem(0);
    XModel * model = (XModel*)args->GetItem(1);
    XOptimizer * optimizer = (XOptimizer*)args->GetItem(2);

    updater->UpdateModel(model, optimizer, SLEEP_TIME_IN_MODEL_UPDATE);

    //fprintf(stderr, "update 1\n");
}

/* 
add a new job of model update (for a parameter) 
>> model - the model that we want to update (on the server side)
>> members - models that would share the updated parameters
>> pid - the parameter index
>> optimizer - the optimizer
>> broadcaster - the worker that would broadcast the new parameter to members
*/
bool XWorkerUpdate::AddJobUpdateSingle(XModel * model, XList * members, int pid,
                                       XOptimizer * optimizer, XWorkerBroadcast * broadcaster)
{
    CheckNTErrors(model != NULL, "No input model!");
    CheckNTErrors(members != NULL, "No member model list!");
    CheckNTErrors(optimizer != NULL, "No optimizer!");
    CheckNTErrors(broadcaster != NULL, "No broadcaster!");
    CheckNTErrors(pid >= 0 && pid < model->paramNum, "Illegal parameter index!");

    XList args;
    args.Add(this);
    args.Add(model);
    args.AddInt(members->count);
    args.AddList(members);
    args.AddInt(pid);
    args.Add(optimizer);
    args.Add(broadcaster);

    if (isInstantRun)
        XWorkerUpdate::UpdateSingle(&args);
    else
        queue.EnqueueJob((void*)(char*)XWorkerUpdate::UpdateSingle, &args);

    return true;
}

/* 
add a new job of model update
>> model - the model that we want to update
>> optimizer - the optimizer
*/
bool XWorkerUpdate::AddJobUpdate(XModel * model, XOptimizer * optimizer)
{
    CheckNTErrors(model != NULL, "No input model!");
    CheckNTErrors(optimizer != NULL, "No optimizer!");

    XList args;
    args.Add(this);
    args.Add(model);
    args.Add(optimizer);

    if(isInstantRun)
        XWorkerUpdate::Update(&args);
    else
        queue.EnqueueJob((void*)(char*)XWorkerUpdate::Update, &args);
    
    return true;
}

}
