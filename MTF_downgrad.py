# -*- coding: utf-8 -*-
"""
Code Reference: https://github.com/sergiovitale/pansharpening-cnn-python-version/blob/master/downgrade_images.py
"""
import numpy as np
import scipy.ndimage as snd
# import scipy.misc as misc
import cv2
from others import gaussian2d, kaiser2d
# from fir_filter_wind import fir_filter_wind

def fir_filter_wind(Hd,w):
    """
	compute fir filter with window method
	Hd: 	desired freqeuncy response (2D)
	w: 		window (2D)
	"""
	
    hd=np.rot90(np.fft.fftshift(np.rot90(Hd,2)),2)
    h=np.fft.fftshift(np.fft.ifft2(hd))
    h=np.rot90(h,2)
    h=h*w
    h=h/np.sum(h)
    
    return h
def downgrade_images(I_MS,I_PAN,ratio,sensor):
    """
    downgrade MS and PAN by a ratio factor with given sensor's gains
    """
    I_MS=np.double(I_MS)
    I_PAN=np.double(I_PAN)
    ratio=np.double(ratio)
    flag_PAN_MTF=0

    if sensor=='QB':
        flag_resize_new = 2
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22],dtype='float32')    # Band Order: B,G,R,NIR
        GNyqPan = 0.15
    elif sensor=='IKONOS':
        flag_resize_new = 2             #MTF usage
        GNyq = np.asarray([0.26,0.28,0.29,0.28],dtype='float32')    # Band Order: B,G,R,NIR
        GNyqPan = 0.17
    elif sensor=='GE1':
        flag_resize_new = 2             # MTF usage
        GNyq = np.asarray([0.23,0.23,0.23,0.23],dtype='float32')    # Band Order: B,G,R,NIR
        GNyqPan = 0.16     
    elif sensor=='WV2':
        flag_resize_new = 2             # MTF usage
        GNyq = [0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.27]
        GNyqPan = 0.11
    elif sensor=='WV3':
        flag_resize_new = 2             #MTF usage
        GNyq = 0.29 * np.ones(8)
        GNyqPan = 0.15
    else:
        flag_resize_new = 1
        
    if flag_resize_new == 1:
        I_MS_LP = np.zeros((I_MS.shape[0],int(np.round(I_MS.shape[1]/ratio)+ratio),int(np.round(I_MS.shape[2]/ratio)+ratio)))
            
        for idim in range(I_MS.shape[0]):
            imslp_pad=np.pad(I_MS[idim,:,:],int(2*ratio),'symmetric')
            # I_MS_LP[idim,:,:]=misc.imresize(imslp_pad,1/ratio,'bicubic',mode='F')
            I_MS_LP[idim,:,:]=cv2.resize(imslp_pad, dsize=(int(imslp_pad.shape[0] // ratio), int(imslp_pad.shape[1] // ratio)), interpolation=cv2.INTER_CUBIC)

        I_MS_LR = I_MS_LP[:,2:-2,2:-2]
       
        I_PAN_pad=np.pad(I_PAN,int(2*ratio),'symmetric')
        # I_PAN_LR=misc.imresize(I_PAN_pad,1/ratio,'bicubic',mode='F')
        I_PAN_LR=cv2.resize(I_PAN_pad, dsize=(int(I_PAN_pad.shape[0] // ratio), int(I_PAN_pad.shape[1] // ratio)), interpolation=cv2.INTER_CUBIC)

        I_PAN_LR=I_PAN_LR[2:-2,2:-2]
        
    elif flag_resize_new==2:
        
        N=41
        I_MS_LP=np.zeros(I_MS.shape)
        fcut=1/ratio
        
        for j in range(I_MS.shape[0]):
            #fir filter with window method
            alpha = np.sqrt(((N-1)*(fcut/2))**2/(-2*np.log(GNyq[j])))
            H=gaussian2d(N,alpha)
            Hd=H/np.max(H)
            w=kaiser2d(N,0.5)
            h=fir_filter_wind(Hd,w)
            I_MS_LP[j,:,:] = snd.filters.correlate(I_MS[j,:,:],np.real(h),mode='nearest')
        
        if flag_PAN_MTF==1:
            #fir filter with window method
            alpha = np.sqrt(((N-1)*(fcut/2))**2/(-2*np.log(GNyqPan)))
            H=gaussian2d(N,alpha)
            Hd=H/np.max(H)
            h=fir_filter_wind(Hd,w)
            I_PAN = snd.filters.correlate(I_PAN,np.real(h),mode='nearest')
            I_PAN_LR=I_PAN[int(ratio/2):-1:int(ratio),int(ratio/2):-1:int(ratio)]
            
        else:
            #bicubic resize
            I_PAN_pad=np.pad(I_PAN,int(2*ratio),'symmetric')
            # I_PAN_LR=misc.imresize(I_PAN_pad,1/ratio,'bicubic',mode='F')
            #     H, W, _ = img.shape
            # output_size = (H // ratio, W // ratio)
            dsize=(int(I_PAN_pad.shape[0] // ratio), int(I_PAN_pad.shape[1] // ratio))
            # dsize = (516,516)
            I_PAN_LR=cv2.resize(I_PAN_pad, dsize, interpolation=cv2.INTER_CUBIC)
            I_PAN_LR=I_PAN_LR[2:-2,2:-2]
            
        I_MS_LR=I_MS_LP[:,int(ratio/2):-1:int(ratio),int(ratio/2):-1:int(ratio)]        
        
    return I_MS_LR,I_PAN_LR
            
         