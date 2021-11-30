/* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2017, Natural Language Processing Lab, Northeastern University.
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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2019-04-24
* I'll attend several conferences and workshops in the following weeks -
* busy days :(
*/

#ifndef __MASK_CUH__
#define __MASK_CUH__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* mask entries of a given tensor (cuda version) */
void _CudaMask(const XTensor * a, const XTensor * mask, XTensor * c = NULL, DTYPE alpha = (DTYPE)1.0);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __MASK_CUH__