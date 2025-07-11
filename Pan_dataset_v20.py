import scipy.io as io
import numpy as np

from Pan_mtfresize import interp23tap_torch, resize_images
from torch.utils.data import  Dataset


"""
getimgblock3():对给定多光谱图像，进行分块并排序，每块包含全部波段，根据idx索引，取到指定分块数组,并增加了重叠图像块功能
    arr: 需要执行padding操作的数组
    partrow: 每块行数
    partcol：每块列数
    idx: 序号

"""
def getimgblock(arr, idx, partrow, partcol, overlap = 0):
    band, r, c = arr.shape
    rnum = r / partrow
    cnum = c / partcol
    tem = idx
    idr = int(tem // cnum)
    idc = int(tem % cnum)
    idrstart = partrow * idr
    idrend = partrow * idr + partrow
    idcstart = partcol * idc
    idcend = partcol * idc + partcol

    if (idrstart - overlap) >= 0:
        idrstart-=overlap

    idrend+= overlap
    if (idcstart-overlap) >= 0:
        idcstart -=overlap

    idcend += overlap
    img= arr[:, idrstart:idrend, idcstart:idcend]
    return img
"""
padding，根据快行数和列数自动计算需要paddingd的行数和列数
    arr: 需要执行padding操作的数组
    partrow: 每块行数
    partcol：每块列数
"""
def padding(arr, partrow, partcol):
    band, r, c = arr.shape
    # print("padding before %s"%str(arr.shape))
    if r % partrow == 0:
        row = r
    else:
        row = r + (partrow - r % partrow)
    if c % partcol == 0:
        col = c
    else:
        col = c + (partcol - c % partcol)
    rowp = row - r
    colp = col - c
    arr = np.pad(arr, ((0, 0), (0, rowp), (0, colp)), "constant")
    # print("padding after %s"%str(arr.shape))
    return arr

class MyDataset(Dataset):
    # def __init__(self, data_path, rrdata_path, blocksize):
    def __init__(self, blocksize,MS,MS_LR, PAN,GT=None, sensor=None):

        if len(MS.shape) == 4:
            if GT is None:
                self.MS = MS[0,:,:,:]
                self.MS_LR = MS_LR[0,:,:,:]
                self.PAN = PAN[0,:,:,:]
                self.GT= None
            else:
                self.MS = MS[0,:,:,:]
                self.MS_LR = MS_LR[0,:,:,:]
                self.PAN = PAN[0,:,:,:]
                self.GT = GT[0,:,:,:]
            
        else:
            if GT is None:
                self.MS = MS
                self.MS_LR = MS_LR
                self.PAN = PAN
                self.GT= None
            else:
                self.MS = MS
                self.MS_LR = MS_LR
                self.PAN = PAN
                self.GT = GT

        self.blocksize = blocksize

        self.band, self.row, self.col = self.MS.shape


    def __len__(self):
        rnum = self.row / self.blocksize
        cnum = self.col / self.blocksize
        num = int(rnum * cnum)
        return num

    def __getitem__(self, idx):
        # print("idx:%s"%idx)
        ms = getimgblock(self.MS,idx,self.blocksize,self.blocksize)
        mslr =getimgblock(self.MS_LR,idx,self.blocksize//4,self.blocksize//4)
        pan = getimgblock(self.PAN,idx,self.blocksize,self.blocksize)

        if self.GT is None:
            output = {
                "idx":idx,
                "ms": ms,
                "mslr": mslr, # rrgt
                "pan": pan,
                'gt':0,
                "band": self.band,
                'H':self.row,
                'W':self.col,
                    }
        else:
            gt =  getimgblock(self.GT,idx,self.blocksize,self.blocksize)      
            output = {
                "idx":idx,
                "ms": ms,
                'gt': gt,
                "mslr": mslr, # rrgt
                "pan": pan,
                "band": self.band,
                'H':self.row,
                'W':self.col,

                    }
        return output



class My_train_Dataset(Dataset):
    def __init__(self, blocksize, MS, MS_LR, PAN, mod, GT = None, sensor=None,device='cpu'):


        # Input: MS:[1,4,512,512], MS_LR:[1,4,128,128], PAN:[1,1,512,512]
        # ration: [4,2,1,0]
        #       MS_LR                   PAN
        #   4: [4,128,128]->[4,32,32], [1,512,512]->[1,64,64], gt [4,64,64]
        #   2: [4,128,128]->[4,64,64], [1,512,512]->[1,128,128], gt [4,128,128]
        #   1: [4,128,128]->[4,128,128], [1,512,512]->[1,256,256], gt None
        #   0: [4,128,128], [1,512,512], gt None

      
        self.blocksize = blocksize

        self.mslr= MS_LR[0].numpy()


        mslrtem = self.mslr.transpose((1,2,0))

        self.mslr2 = interp23tap_torch(mslrtem, 2, device)
        self.mslr2 = self.mslr2.transpose((2,0,1))

        self.mslr4 = interp23tap_torch(mslrtem, 4, device)
        self.mslr4 = self.mslr4.transpose((2,0,1))

        # if ration in [2,1]:
        #     _, self.pan = resize_images(MS_LR[0].numpy(), PAN[0].numpy(),ration,sensor)
        # else:
        #     self.pan =  PAN[0].numpy()
        _, self.pan2 = resize_images(MS_LR[0].numpy(), PAN[0].numpy(),2,sensor)
        _, self.pan1 = resize_images(MS_LR[0].numpy(), PAN[0].numpy(),1,sensor)
        self.pan =  PAN[0].numpy()
        self.mod = mod
        if mod == "FR":
            self.GT = 0
        else:
            self.GT =GT[0].numpy()
        self.band, self.row, self.col = self.mslr.shape

        
        print("............")

    def __len__(self):
        if self.row > self.blocksize:
            n = self.row//self.blocksize
            num = int(n * n)
        else:
            num = 1
        return num

    def __getitem__(self, idx):

        if  self.row > self.blocksize:
            # print("idx:%s"%idx)
           
            mslr =getimgblock(self.mslr,idx,self.blocksize,self.blocksize)
            mslr2 =getimgblock(self.mslr2,idx,self.blocksize*2,self.blocksize*2)
            mslr4 =getimgblock(self.mslr4,idx,self.blocksize*4,self.blocksize*4)

            # if self.ration == 0:
            #     mslr =getimgblock(self.mslr,idx,self.blocksize//4,self.blocksize//4)
            # else:
            #     mslr =getimgblock(self.mslr,idx,self.blocksize//2,self.blocksize//2)
            
            pan2 = getimgblock(self.pan2,idx,self.blocksize,self.blocksize)
            pan1 = getimgblock(self.pan1,idx,self.blocksize*2,self.blocksize*2)
            pan = getimgblock(self.pan,idx,self.blocksize*4,self.blocksize*4)
            # if self.ration in [1,0]:
            #     gt = 2
            # else:
            #     gt = getimgblock(self.gt,idx,self.blocksize,self.blocksize)

        else:
            mslr = self.mslr
            pan = self.pan
            pan1 = self.pan1
            pan2 = self.pan2

        if  self.mod =='FR':
            output = {
                "idx":idx,
                "mslr": mslr, 
                "mslr2": mslr2,
                "mslr4": mslr4,
                "pan": pan,
                "pan2": pan2,
                "pan1": pan1,
                "band": self.band,
                'H':self.row,
                'W':self.col,
                    }
        else:
            gt =getimgblock(self.GT,idx,self.blocksize*4,self.blocksize*4)
            output = {
                "idx":idx,
                "mslr": mslr, 
                "mslr2": mslr2,
                "mslr4": mslr4,
                "pan": pan,
                "pan2": pan2,
                "pan1": pan1,
                "band": self.band,
                'gt':gt,
                'H':self.row,
                'W':self.col,
                    }

        return output



import torch
if __name__ == "__main__":
    # path = r'D:\documents\datasets\PAirMax\GE_Lond_Urb\FR\GE_Lond_Urb_GE_FR.mat'
    # rrdata_path = r'D:\documents\datasets\PAirMax\GE_Lond_Urb\RR\GE_Lond_Urb_GE_RR.mat'
    # blocksize = 64

    mslr = torch.randn(1, 4, 512, 512)
    ms = torch.randn(1, 4, 2048, 2048)
    pan = torch.randn(1, 1, 2048, 2048)

    rations = [2,1,0]
    sensor = 'WV4'
    for i in rations:
        datasets = My_train_Dataset(ms,mslr,pan,i,64,sensor,device='cpu')
        number = datasets.__len__()
        id1 = datasets.__getitem__(2)
    print('aaaaaaaaaaaaaa')