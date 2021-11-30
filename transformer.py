import sys
import time
import math
import NiuTensor as niu

class Encoder:
    def __init__(self):
        self.layers=tt.GetEncoderLayers(model)
    def forward(self,batchEnc,paddingEnc):
        mask=tt.GetMask(model,paddingEnc)
        x=tt.Embedding_En(model,batchEnc,False,True)
        dropoutP=tt.GetEncoderDropoutP(model)
        if trainning and dropoutP>0:
                x=tt.Dropout(x,dropoutP)
        for i in range(0,self.layers):
            attnBefore=tt.EncLayerNorm_att(x,model,i,True,False)
            att=tt.EncoderAttn(model,attnBefore,attnBefore,attnBefore,mask,trainning,i)
            dropoutP=tt.GetEncoderDropoutP(model)
            if trainning and dropoutP>0:
                att=tt.Dropout(att,dropoutP)
            res = niu.Sum(att,x)
            attnAfter = tt.EncLayerNorm_att(res, model, i, False, True)
            fnnBefore = tt.EncLayerNorm_fnn(attnAfter, model,i, True, False)
            fnn = tt.FNNMake_En(model,i,fnnBefore,trainning)
            dropoutP=tt.GetEncoderDropoutP(model)
            if trainning and dropoutP>0:
                fnn = tt.Dropout(fnn,dropoutP);
            res = niu.Sum(fnn,attnAfter)
            x = tt.EncLayerNorm_fnn(res,model,i,False,True)
        if model.encoder.preNorm>0:
            x = tt.LayerNorm_encoder(x,model)
        return x
class Decoder:
    def __init__(self):
        self.layers=tt.GetDecoderLayers(model)
    def forward(self,batchDec,paddingEnc,paddingDec,encoding):
        maskDec = niu.XTensor()
        maskEncDec = niu.XTensor()
        model.MakeMTMaskDec(paddingEnc,paddingDec,maskDec, maskEncDec)
        x = tt.Embedding_De(model,batchDec,True,trainning,batchDec.GetDim(1))
        dropoutP=tt.GetDecoderDropoutP(model)
        if trainning and dropoutP>0:
            x=tt.Dropout(x,dropoutP)
        for i in range(0,self.layers):
            selfAttnBefore = tt.DecLayerNorm_att(x, model,i, True, False)
            att = tt.DecoderAttn(model,selfAttnBefore,selfAttnBefore,selfAttnBefore,maskDec,trainning,i)
            dropoutP=tt.GetDecoderDropoutP(model)
            if trainning and dropoutP>0:
                att=tt.Dropout(att,dropoutP)
            res=niu.Sum(att,x)
            selfAttnAfter = tt.DecLayerNorm_att(res, model, i, False, True)
            endeAttnBefore = tt.EnDecLayerNorm_att(selfAttnAfter,model,i,True,False)
            ende = tt.EnDecoderAttn(model,encoding,endeAttnBefore,encoding,maskEncDec,trainning,i)
            dropoutP=tt.GetDecoderDropoutP(model)
            if trainning and dropoutP>0:
                ende=tt.Dropout(ende,dropoutP)
            res = niu.Sum(ende,selfAttnAfter)
            endeAttnAfter = tt.EnDecLayerNorm_att(res,model,i,False,True)
            fnnBefore = tt.DecLayerNorm_fnn(endeAttnAfter,model,i,True,False)
            fnn = tt.FNNMake_De(model,i,fnnBefore,trainning)
            dropoutP=tt.GetDecoderDropoutP(model)
            if trainning and dropoutP>0:
                fnn=tt.Dropout(fnn,dropoutP)
            res = niu.Sum(fnn,endeAttnAfter)
            x = tt.DecLayerNorm_fnn(res,model,i,False,True)
        if model.decoder.preNorm>0:
            x = tt.LayerNorm_decoder(x,model)
        return x
            
            
class Transformers:
    def forward(self):
        encoder_t = Encoder()
        decoder_t = Decoder()
        gradStep = 0
        wordCounttotal = 0
        batchCounttotal = 0
        step = 0
        validStep = 0
        nSkipped = 0
        isEnd = False
        nStepCheck = 0
        nCheckpoint = 0
        
        trainer.PrepareModel(model)
        startt = time.process_time()
        tt.Init_batchLoader(trainer,config_dict["trainFN"],True)
        
        for epoch in range(0,trainer.nepoch):
            wordCount = 0
            loss = 0
            
            trainer.batchLoader.ClearBuf()
            while trainer.batchLoader.IsEmpty()==False:
                net = niu.XNet()
                net.Clear()
                batchEnc = niu.XTensor()
                batchDec = niu.XTensor()
                label = niu.XTensor()
                paddingEnc = niu.XTensor()
                paddingDec = niu.XTensor()
                wc,ws = tt.LoadBatch(trainer,batchEnc,paddingEnc,batchDec,paddingDec,label,trainer.sBatchSize,trainer.wBatchSize,model.devID)
                
                output = niu.XTensor()
                if model.isLM:
                    encoding = encoder_t.forward(batchEnc,paddingEnc)
                    tt.MakeOutputLayer(model,encoding, output, True, True)
                elif model.isMT:
                    encoding = encoder_t.forward(batchEnc,paddingEnc)
                    decoding = decoder_t.forward(batchDec,paddingEnc,paddingDec,encoding)
                    tt.MakeOutputLayer(model,decoding, output, True, True)
                labelOnehot = tt.IndexToOnehot(trainer,label)
                lossTensor = niu.CrossEntropy_trans(output, labelOnehot, paddingDec)
                lossBatch = niu.ReduceSumAllValue(lossTensor)
                lossLocal = lossBatch/wc
                doUpdate = niu.Check_doUpdate(lossLocal)
                if doUpdate:
                    net.Backward(lossTensor)
                    gradStep += 1
                    loss += lossBatch
                    wordCount += wc
                    wordCounttotal += wc
                    batchCounttotal += ws
                    if gradStep == trainer.updateStep:
                        warmupEndLR = trainer.lrate
                        warmupInitLR = 1e-7
                        lrStep = (warmupEndLR - warmupInitLR) / trainer.nwarmup
                        decayFactor = warmupEndLR * pow(trainer.nwarmup, 0.5)
                        
                        if step<trainer.nwarmup:
                            lr = warmupInitLR + step * lrStep
                        else:
                            lr = decayFactor * pow(step, -0.5)
                        trainer.Update(model,lr)
                        gradStep = 0
                        validStep+=1
                else:
                    nSkipped+=1
                step+=1
                if step>=trainer.nstep:
                    isEnd = True
                    break
                if step%100==0:
                    elapsed=time.process_time()-startt
                    print("elapsed=%.1fs, step=%d, epoch=%d, total word=%d, total batch=%d, loss=%.3f, ppl=%.3f, lr=%.2e"%(elapsed, step, epoch, wordCounttotal, batchCounttotal,loss / wordCount / math.log(2.0), math.exp(loss / wordCount), lr))
                #elapsed=time.process_time()-startt
                #print("elapsed=%.1fs, step=%d, epoch=%d, total word=%d, total batch=%d, loss=%.3f, ppl=%.3f, lr=%.2e"%(elapsed, step, epoch, wordCounttotal, batchCounttotal,loss / wordCount / math.log(2.0), math.exp(loss / wordCount), lr))
                nStepCheck+=1
                if trainer.nStepCheckpoint>0 and nStepCheck>=trainer.nStepCheckpoint:
                    tt.MakeCheckpoint(trainer,config,model,"step",step)
                    nStepCheck = 0
                    nCheckpoint+=1
            if isEnd:
                break
            if trainer.useEpochCheckpoint:
                tt.MakeCheckpoint(trainer,config,model,"epoch",epoch)
        elapsed = time.process_time() - startt
        epoch = min(epoch,trainer.nepoch)
        print("lr=%.2e, elapsed=%.1fs, step=%d, epoch=%d, word=%d, loss=%.3f, ppl=%.3f"%(lr, elapsed, step, epoch, wordCounttotal, loss / wordCount / math.log(2.0), math.exp(loss / wordCount)))
        print("training finished (took %.1fs, step=%d, skipped=%d and epoch=%d)"%(elapsed, step, nSkipped, epoch))
        print("saving the final model")
        tt.Dump(model,config_dict["modelFN"])
        
        
        
        

if __name__ == '__main__':
    tt=niu.Transformer_py()
    config=tt.ConfigInit(len(sys.argv[1:]),sys.argv[1:])
    config_dict=tt.Config_dict(config)
    trainning=True
    if config_dict["trainFN"]!="":
        model=niu.Model()
        model.InitModel(config)
        tt.Cache_disable(model)
        trainer=niu.Trainer()
        trainer.Init(config)
        transformer = Transformers()
        transformer.forward()
        #trainer.Train(trainer,config,model)
    
    if config_dict["testFN"]!="" and config_dict["outputFN"]!="":
        tt.Translate_Fun(config)
        
        
    