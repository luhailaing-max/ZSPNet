# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
import os
import torch.optim as optim
import Pan_losses
import scipy.io as io
import time
import sys
# sys.path.append(r'D:\codes\pansharpening\MyPNN')
# sys.path.append(r'D:\codes\pansharpening\MyPNN\models')

# sys.path.append(r'/home/jmhaut/Hailiang/codes/pansharpening/MyPNN')
# sys.path.append(r'/home/jmhaut/Hailiang/codes/pansharpening/MyPNN/models')
from Pan_model_v20 import modelv20
import torch.nn.functional as F
# from pcgrad import PCGrad
from Pan_dataset_v20 import MyDataset, padding,My_train_Dataset
from torch.utils.data import DataLoader
from Pan_mtfresize import resize_images, DowngradeProtocol, gen_mtf,upgrade

def masked(arr,reference):
    import copy
    mask=copy.deepcopy(arr[:,:,:])
    mask1 = copy.deepcopy(arr[:,:,:])
    # coh[coh>throd] = throd

    mask[mask<2047]= 1
    mask[mask>2047] = 0


    mask1[mask1<2047]= 0
    mask1[mask1>2047] = 1

    # for i in range(arr.shape[2]):
    #     arr[:,:,i] = arr[:,:,i]*mask
    # for i in range(arr.shape[2]):
    #     reference[:,:,i] = reference[:,:,i]*mask1    
    reference = reference*mask1
    arr = arr * mask
    arr = arr + reference

    return arr

def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    np.random.seed(seed) # Numpy module.
    random.seed(seed) # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def Gradnorm(model,task_loss,t,optimizer,initial_task_loss):
    # compute the weighted loss w_i(t) * L_i(t)
      
        weighted_task_loss = torch.mul(model.module.weights, task_loss)
        # initialize the initial loss L(0) if t=0
     
        # get the total loss
        loss = torch.sum(weighted_task_loss)
        # clear the gradients
        optimizer.zero_grad()
        # do the backward pass to compute the gradients for the whole set of weights
        # This is equivalent to compute each \nabla_W L_i(t)
        loss.backward(retain_graph=True)

        # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
        #print('Before turning to 0: {}'.format(model.weights.grad))
        model.module.weights.grad.data = model.module.weights.grad.data * 0.0
        #print('Turning to 0: {}'.format(model.weights.grad))


        # switch for each weighting algorithm:
        # --> grad norm

            
        # get layer of shared weights
        # W = model.module.get_selected_layer()
        W = model
        # get the gradient norms for each of the tasks
        # G^{(i)}_w(t) 
        norms = []
        for i in range(len(task_loss)):
            # get the gradient of this task loss with respect to the shared parameters
            gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
            # compute the norm
            norms.append(torch.norm(torch.mul(model.module.weights[i], gygw[0])))
        norms = torch.stack(norms)
        #print('G_w(t): {}'.format(norms))


        # compute the inverse training rate r_i(t) 
        # \curl{L}_i 
        if torch.cuda.is_available():
            loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
        else:
            loss_ratio = task_loss.data.numpy() / initial_task_loss
        # r_i(t)
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)
        #print('r_i(t): {}'.format(inverse_train_rate))


        # compute the mean norm \tilde{G}_w(t) 
        if torch.cuda.is_available():
            mean_norm = np.mean(norms.data.cpu().numpy())
        else:
            mean_norm = np.mean(norms.data.numpy())
        #print('tilde G_w(t): {}'.format(mean_norm))


        # compute the GradNorm loss 
        # this term has to remain constant
        constant_term = torch.tensor(mean_norm * (inverse_train_rate ** 0.12), requires_grad=False)
        if torch.cuda.is_available():
            constant_term = constant_term.cuda()
        #print('Constant term: {}'.format(constant_term))
        # this is the GradNorm loss itself
        grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
        #print('GradNorm loss {}'.format(grad_norm_loss))

        # compute the gradient for the weights
        model.module.weights.grad = torch.autograd.grad(grad_norm_loss, model.module.weights)[0]


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, pathdir='./',  path='checkpoint.pth', trace_func=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.pathdir = pathdir
        self.trace_func = trace_func
    def __call__(self, val_loss, egrasloss,model,mod):

        if mod == 'FR':
            self.save_checkpoint(val_loss, egrasloss,model)
        else:
            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss,egrasloss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                # print('    EarlyStopping counter: %s, out of %s'%(self.counter,self.patience))
                if self.counter >= self.patience:
                    self.early_stop = True
                    self.counter =0
            else:
                self.best_score = score
                
                self.save_checkpoint(val_loss, egrasloss,model)
                self.counter = 0

    def save_checkpoint(self, val_loss,egrasloss, model):
        '''Saves model when validation loss decrease.'''
        if not os.path.exists(self.pathdir):
               os.makedirs(self.pathdir)
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print('-- Loss decreased (%s --> %s). Ergas loss: %s,  Saved best model in %s ...'%(self.val_loss_min,val_loss,egrasloss,self.path))

        self.val_loss_min = val_loss


def train_step(
        model,data,blocksize,epochs,batch_size,ratio,
        sensor,lr,weight_decay,
        checkpoint_path,filename,nameidx,mod,logfre,device,ckpdir):

    pathname = filename[nameidx]+'_'+mod+'_best.pth'
    checkpoint_dir =  os.path.join(checkpoint_path, pathname)
    # print(model)
    if os.path.exists(ckpdir):
        # model.load_state_dict(torch.load(ckpdir, map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(ckpdir, map_location=torch.device(device)))
        # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(ckpdir, map_location=torch.device('cpu')).items()})
        # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(ckpdir).items()})
        print("    Success to loading model dict from %s ....."%ckpdir)

    else:
        print("    Failed to load model dict  from %s ....."%ckpdir)

    # Loss for supervised part
    if mod == 'RR':
        lossMSE = torch.nn.MSELoss().to(device)
    else:
    # Loss for unsupervised part
        # downgrade = DowngradeProtocol(gen_mtf(ratio, sensor), ratio, device).to(device)
        LStruct = Pan_losses.StructuralLoss(ratio, device)

        loss_reprojected_ergas = Pan_losses.ERGAS(ratio).to(device)
        # loss_reprojected_ergas = torch.nn.L1Loss(reduction='mean')
        # loss_reprojected_ergas =  Pan_losses.SSIM().to(device)
        # loss_reprojected_ergas =  Pan_losses.SID().to(device)
        # loss_reprojected_ergas =  Pan_losses.Loss_SAM().to(device)



        # downgrade1 = DowngradeProtocol(gen_mtf(ratio//2, sensor), ratio//2, device).to(device)
        # LStruct1 = Pan_losses.StructuralLoss(ratio//2, device)
        # loss_reprojected_ergas1 = Pan_losses.ERGAS(ratio//2).to(device)

        # Llambda = Pan_losses.ReproDLambdaKhan(device).to(device)

    MS_ = data["ms"]
    MS_lr = data["mslr"]
    PAN_ = data["pan"]
    if mod == "RR":
        GT = data["gt"]
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        GT = 0
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # pcgradoptimizer = PCGrad(optimizer)

    # optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.99,weight_decay = weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=int(0.1 * epochs[-1]),
    #                                                  threshold_mode='rel', cooldown=int(0.02 * epochs[-1]), min_lr=1e-7,
    #                                                  verbose=True)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # blocksize, MS, MS_LR, PAN, ration, GT = None, sensor=None,device='cpu'

    traindata = My_train_Dataset(blocksize,MS_,MS_lr,PAN_,mod,GT,sensor,device)# 2 is no use
    train_loader = DataLoader(dataset=traindata, batch_size=batch_size[0], shuffle=True)

    model.train()

    r = torch.tensor([0])
    c = torch.tensor([0])
    r = r.repeat(batch_size, 1)
    c = c.repeat(batch_size, 1)

    early_stopping = EarlyStopping(patience=41, verbose=False, pathdir = checkpoint_path, path=checkpoint_dir)

    # temdata = traindata.__getitem__(0)
    # if rations[i] ==2:
    #     print("Ration: %s. Performing supervised learning part, band:%s, input:%s pan:%s ....."%(rations[i],temdata['band'], temdata['inp'].shape, temdata['pan2'].shape))
    # elif rations[i] ==1:
    #     print("Ration: %s. Performing unsupervised learning part, band:%s, input:%s pan:%s ....."%(rations[i], temdata['band'], temdata['inp'].shape, temdata['pan1'].shape))
    # else: 
    #     print("Ration: %s. Performing unsupervised learning part, band:%s, input:%s pan:%s ....."%(rations[i], temdata['band'], temdata['inp'].shape, temdata['pan'].shape))
    
    # if rations[i] in [0]:
    #     alpha = 0.01

    for epoch in range(0,epochs):
        # losses = []
        lossesstruc=[0.]
        
        lossergas = [0.]
        lossmse = [0.]
        times = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))    
        # print("    %s: Epoch %s/%s......."%(times,epoch, epochs))
        
        for _, datai in enumerate(train_loader):

            optimizer.zero_grad()  
            pan1 = datai["pan1"].to(device).type(torch.float32)
            pan2 = datai["pan2"].to(device).type(torch.float32)
            mslr = datai["mslr"].to(device).type(torch.float32)
            mslr2 = datai["mslr2"].to(device).type(torch.float32)
            mslr4 = datai["mslr4"].to(device).type(torch.float32)
            pan = datai["pan"].to(device).type(torch.float32)
        
            # from fvcore.nn import FlopCountAnalysis, parameter_count
            # # flops = FlopCountAnalysis(model, (mslr,pan,pan1,pan2,mslr2,mslr4))
            # # print(f"FLOPs: {flops.total() / 1e6:.2f} MFLOPs")

            # # # 统计参数量
            # params = parameter_count(model)
            # print(f"参数量: {params[''] / 1.0:.2f} ")
            # return

            # out1,out2,out3,out= model(inp,pan,pan1,pan2)
            out1,out2,out3= model(mslr,pan,pan1,pan2,mslr2,mslr4)

            # bb,cc,rr,ww= ms.shape
            if mod == "RR" :
                gt = datai["gt"].to(device).type(torch.float32)
                loss_mse = lossMSE(out3, gt)
                loss_mse.backward()
                lossmse.append(loss_mse.item())
                optimizer.step()
            else:
                # downgraded_shifted_outputs = downgrade(out3, r, c).type(torch.float32)
                downgraded_shifted_outputs = torch.nn.functional.interpolate(out3,scale_factor=1/4,mode="bilinear")
                loss_ergas = loss_reprojected_ergas(downgraded_shifted_outputs, mslr)    
                loss_struct, _ = LStruct(out3, pan) 

                # loss_mse = lossmse(downgraded_shifted_outputs ,mslr)
                
                # loss =  loss_ergas*alpha + loss_struct + loss_mse
                lossesstruc.append(loss_struct.item())
                lossergas.append(loss_ergas.item())

                # pcgradoptimizer.pc_backward([(loss_ergas*alpha).type(torch.float32), loss_struct.type(torch.float32)])
                # pcgradoptimizer.pc_backward([loss_ergas.type(torch.float32), loss_struct.type(torch.float32)])
                # loss = loss_ergas + loss_struct
                # loss.backward()

                          ## Gradnorm
                losses = [loss_ergas, loss_struct]
                losses = torch.stack(losses)   
                if epoch == 0:
                    # set L(0)
                    if torch.cuda.is_available():
                        initial_task_loss = losses.data.cpu()
                    else:
                        initial_task_loss = losses.data
                    initial_task_loss = initial_task_loss.numpy()

                Gradnorm(model,losses,epoch,optimizer,initial_task_loss)
                optimizer.step()



 

        lossmse_avg = sum(lossmse)/len(lossmse)
        lossesstruc_avg = sum(lossesstruc)/len(lossesstruc)
        lossergas_avg = sum(lossergas)/len(lossergas)

        if (epoch + 1) % logfre == 0:
            # print('    %s: Epoch %s/%s ==> loss: %s,Stru loss: %s, ERGAS loss: %s, MSE %s ............'%(times,epoch+1, epochs[i], loss.item(), lossesstruc,lossergas,loss_mse))
            print('    %s: Epoch %s/%s ==> Stru loss: %s, ERGAS loss: %s, MSE loss: %s ............'%(times,epoch+1, epochs, lossesstruc_avg,lossergas_avg,lossmse_avg))

        if mod == 'RR':
            early_stopping(lossmse_avg,lossergas_avg, model,mod)
        else:
            early_stopping(lossesstruc_avg,lossergas_avg, model,mod)
        # if lossesstruc < 0.01:
        #     return
        if early_stopping.counter == 40:
            scheduler.step()
            lr = lr*0.8
            # alpha = alpha *0.9
            print("scheduler step---lr: %s------------------------------"%(lr))

        # if early_stopping.early_stop:
        #     print("Early stopping on epoch:%d....."%(epoch+1))
        #     break

    return 

def val_step(model,data,blocksize,checkpoint_dir,batch_size,mod,device,sensor):
   
    MS_ = data["ms"]
    MS_lr = data["mslr"]
    PAN_ = data["pan"]
    GT = data["gt"]

    traindata = My_train_Dataset(blocksize,MS_,MS_lr,PAN_,mod,GT,sensor,device)
    train_loader = DataLoader(dataset=traindata, batch_size=batch_size[-1], shuffle=True)

    # tem= traindata.__getitem__(1)
    _, C, H, W = MS_.shape

    if os.path.exists(checkpoint_dir):
        # model.load_state_dict(torch.load(checkpoint_dir, map_location=torch.device('cpu')))
        # model.load_state_dict(torch.load(checkpoint_dir, map_location=torch.device(device)))
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(checkpoint_dir, map_location=torch.device('cpu')).items()})
        # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(checkpoint_dir).items()})
        print("    Success to loading model dict from %s ....."%checkpoint_dir)

    else:
        print("    Stoping, failed to load model dict  from %s ....."%checkpoint_dir)
        return None
    
    model.eval()
    with torch.no_grad():
        img = torch.zeros((C,H,W))
        for _, dataval in enumerate(train_loader):
            idxx = dataval["idx"].to(device)
            mslr = dataval["mslr"].to(device)
            mslr2=dataval["mslr2"].to(device)
            mslr4=dataval["mslr4"].to(device)
            pan2 = dataval["pan2"].to(device)
            pan1 = dataval["pan1"].to(device)
            pan = dataval["pan"].to(device)

            # out1,out2,out3,out= model(inp,pan,pan1,pan2)

          
            out1,out2,out3= model(mslr,pan,pan1,pan2,mslr2,mslr4)
            size = blocksize*4
            for j, da in enumerate(out3):
                    rnum = H //size
                    cnum = W // size
        
                    if batch_size[-1]>1:
                        tem = idxx[j]
                    else:
                        tem = idxx
                    idr = int(tem // cnum)
                    idc = int(tem % cnum)
                    idrstart = size * idr
                    idrend = size * idr + size
                    idcstart = size * idc
                    idcend = size * idc + size
                    img[:, idrstart:idrend, idcstart:idcend] = da

    return img
# 作为填补的日期 （主日期和填补日期必须是不同年份的相同月份）
def main(nameid, plat, timestemps, mod, fetures):

    print(__file__)

    platforme = plat 
    flag= 'train_test' #  train_test
    mod = mod
    nameidx = nameid

    if mod == 'RR':
        min_size = 512
        mlist = [1,1,1]
        batch_size = [64]
        epochs = 1000
        timestemp = "000"
        logfre =20
    else:
        min_size = 2048
        mlist = [1,1,1]
        batch_size = [64] #[256]
        epochs = 0
        timestemp = timestemps[nameid]
        logfre =1

    dim = 64
    blocksize = fetures[0]
    seed = 42
    lr = 0.0001
    weight_decay = 0.000001
   
    ratio = 4

    filename = ['W4_Mexi_Nat_W4','W4_Mexi_Urb_W4','W2_Miam_Mix_W2','W2_Miam_Urb_W2','W3_Muni_Mix_W3','W3_Muni_Nat_W3','W3_Muni_Urb_W3','GE_Lond_Urb_GE','GE_Tren_Urb_GE']
    print('---------------------------Nameid is : --%s--%s-----------------------------------'%(nameidx,filename[nameidx]))
    if nameidx in [0,1]:
        sensor = 'WV4' # WV2, WV3, WV4, GE1
    elif nameidx in [2,3]:
        sensor = 'WV2' # WV2, WV3, WV4, GE1
    elif nameidx in [4,5,6]: 
        sensor = 'WV3' # WV2, WV3, WV4, GE1
    elif  nameidx in [7,8]:
        sensor = 'GE1' # WV2, WV3, WV4, GE1
    else:
        sensor = None

    namemodel = 'modelv20' # modelv20
    modelname ={'modelv20':modelv20 }

    # Load data
    if platforme=='win':
        path = r'E:\codes\PAN\ZSPNet\PairMax\\'+filename[nameidx]+'_'+mod+'.mat'
        temppthpath = r'E:\codes\PAN\\ZSPNet\pth'
        temp_mat_outpath = r'E:\codesPAN\\ZSPNet\outputs'
        times = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        if flag == 'test':
            checkpoint_path =  os.path.join(temppthpath,namemodel,timestemp,filename[nameidx]+'_'+mod+'_best.pth')
            matoutpath =  os.path.join(temp_mat_outpath,namemodel,timestemp)
        else:
            checkpoint_path =  os.path.join(temppthpath,namemodel,str(times))
            matoutpath =  os.path.join(temp_mat_outpath,namemodel,str(times))
    else:
        path = './PairMax/'+filename[nameidx]+'_'+mod+'.mat'
        temppthpath = './pth/'
        temp_mat_outpath = './outputs/'
        times = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        if flag == 'test':
            checkpoint_path =  os.path.join(temppthpath,namemodel,timestemp,filename[nameidx]+'_'+mod+'_best.pth')
            matoutpath =  os.path.join(temp_mat_outpath,namemodel,timestemp)
        else:
            checkpoint_path =  os.path.join(temppthpath,namemodel,str(times))
            matoutpath =  os.path.join(temp_mat_outpath,namemodel,str(times))

    temp = io.loadmat(path)
    ms_temp= temp['I_MS'].astype('float32')/ 2047.0
    MS = ms_temp.transpose((2,0,1))
    ms_lr_temp= temp['I_MS_LR'].astype('float32')/ 2047.0
    MS_LR = ms_lr_temp.transpose((2,0,1))
    pan_temp = temp['I_PAN'].astype('float32')/ 2047.0
    PAN = pan_temp[np.newaxis,:, :]
    C, H,W = MS.shape

    MS = padding(MS,min_size,min_size)
    MS_LR = padding(MS_LR,min_size,min_size)
    PAN = padding(PAN,min_size,min_size)
    c,h,w = MS.shape

    if mod == 'RR':
        GTtem = temp['I_GT'].astype('float32')/ 2047.0
        GT = GTtem.transpose((2,0,1))
        GT = padding(GT,min_size,min_size)
    else:
        GT = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
 
    dataset = MyDataset(min_size,MS,MS_LR, PAN, GT)
    datasetloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    img  = torch.zeros((c,h,w))
    
    model = modelname[namemodel](inc=C,outc=C,fetures=fetures,dim = dim,mlist = mlist)

    # print(model)
    # if torch.cuda.device_count() > 1:
    #     # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
    #     # model = torch.nn.DataParallel(model, device_ids=[1])
    #     model = torch.nn.DataParallel(model)

    model.to(device)
    ckpdir =  os.path.join(temppthpath,namemodel,timestemp,filename[nameidx]+'_RR_best.pth')
    for i, data in enumerate(datasetloader):
        print("Process %s / %s in whole image........"%(i+1,len(datasetloader)))

        if mod == 'RR':
            print("    Performing RR")
            
            train_step(model,data,blocksize,epochs,batch_size,ratio,
                       sensor,lr,weight_decay,
                       checkpoint_path,filename,nameidx,mod,logfre,device,ckpdir)
            
            pathname = filename[nameidx]+'_'+mod+'_best.pth'
            checkpoint_dir =  os.path.join(checkpoint_path, pathname)
            
            outdata =  val_step(model,data,blocksize,checkpoint_dir,batch_size,mod,device,sensor)  
        else:

            print("    Performing FR ......")
       
            if epochs == 0:
                print("FR epochs is 0..... ")
                outdata = val_step(model,data,blocksize,ckpdir,batch_size,mod,device,sensor)
            else:
                train_step(model,data,blocksize,epochs,batch_size,ratio,
                        sensor,lr,weight_decay,
                        checkpoint_path,filename,nameidx,mod,logfre,device,ckpdir)
                
                pathname = filename[nameidx]+'_'+mod+'_best.pth'
                checkpoint_dir =  os.path.join(checkpoint_path, pathname)

                outdata = val_step(model,data,blocksize,checkpoint_dir,batch_size,mod,device,sensor)

        rnum = h //min_size
        cnum = w // min_size
        tem = i
        idr = int(tem // cnum)
        idc = int(tem % cnum)
        idrstart = min_size * idr
        idrend = min_size * idr + min_size
        idcstart = min_size * idc
        idcend = min_size * idc + min_size
        img[:, idrstart:idrend, idcstart:idcend] = outdata  
        # break
    out = img[:, 0:H, 0:W].numpy()
    out = out.transpose((1,2,0))*2047.0

    # 四舍五入
    out = np.round(out).astype(np.uint16)
    temp = io.loadmat(path)
    ms_reference= temp['I_MS']
    out = masked(out,ms_reference).astype(np.uint16)

    if not os.path.exists(matoutpath):
        os.makedirs(matoutpath)

    if flag == 'train_test':
        matoutdir=  os.path.join(matoutpath,'output_'+filename[nameidx]+'_'+mod+'.mat')
    else:
        matoutdir=  os.path.join(matoutpath,'output_'+filename[nameidx]+'_'+mod+'_best.mat')

    io.savemat(matoutdir, {'mypnn_MS': out})

    print("Save successfully in dir:%s......"%matoutdir)


if __name__ == "__main__":
    # filename = ['W4_Mexi_Nat_W4','W4_Mexi_Urb_W4','W2_Miam_Mix_W2','W2_Miam_Urb_W2',
    # 'W3_Muni_Mix_W3','W3_Muni_Nat_W3','W3_Muni_Urb_W3','GE_Lond_Urb_GE','GE_Tren_Urb_GE']

    plat = 'linux' # win, linux
    # v1
    # timestemps = ['202501111020','202501111040','202501111101','202501111121','202501111142','202501111203','202501111223','202501111244','202501111302']
    # v2
    # timestemps = ['202501111340','202501111404','202501111428','202501111451','202501111515','202501111539','202501111603','202501111626','202501111650']
    # v3
    # timestemps = ['202501111824','202501111851','202501111916','202501111944','202501112009','202501112036','202501112101','202501112126','202501112152']

    #vv1
    # timestemps = ['202501121031','202501121045','202501121102','202501121119','202501121135','202501121152','202501121209','202501121225','202501121243']
    # #vv2
    # timestemps = ['202501121519','202501121542','202501121602','202501121624','202501121647','202501121709','202501121732','202501121754','202501121815']
    # #vv3
    # timestemps = ['202501121935','202501122000','202501122024','202501122047','202501122111','202501122135','202501122200','202501122223','202501122247']


    # #vv3-8
    # timestemps = ['202501131201','202501131207','202501131213','202501131218','202501131224','202501131231','202501131238','202501131245','202501131251']

    # #vv3-16
    # timestemps = ['202501131430','202501131434','202501131437','202501131441','202501131445','202501131450','202501131456','202501131500','202501131505']

    # #vv3-32
    # timestemps = ['202501131558','202501131644','202501131726','202501131813','202501131901','202501131950','202501132043','202501132131','202501132215']


    # timestemps = ['000','000','000','000','000','000','000','000','000']
    timestemps = ['000','000','000','000','000','000','000','202504271823','000']
    import time
    time_start = time.time()
    mod = 'RR'  # FR, RR
    fetures=[4,8,16]
    # fetures=[8,16,32]
    # fetures=[16,32,64]
    # fetures=[32,64,128]
    for nameid in [7]: # [0,1,2,3,4,5,6,7,8]:              
        main(nameid, plat, timestemps, mod, fetures)

    Performed_time = time.time() - time_start
    if mod == 'RR' :
        print('Training time: {:.2f} s'.format(Performed_time))
    else:
        print('Testing time: {:.2f} s'.format(Performed_time))
    print("#####finish......######")
