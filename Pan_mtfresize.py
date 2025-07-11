# -*- coding: utf-8 -*-
"""
Code Reference: https://github.com/matciotola/Z-PNN/blob/master/input_prepocessing.py
                https://github.com/matciotola/Lambda-PNN/blob/main/metrics.py
                https://github.com/liangjiandeng/DLPan-Toolbox/blob/main/01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py
"""

import math
import numpy as np
import torch
from skimage import transform
from torch import nn
import torch.nn.functional as fun
import scipy.ndimage.filters as ft


def half_pixel_shift(img, direction, half_kernel, device='cpu'):
    img = img.double()
    nbands = img.shape[1]

    directions = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']
    assert direction in directions, "Error: wrong direction input '{}' - allowed values " \
                                    "are ''N'', ''S'', ''NE''...".format(direction)

    half_kernel = torch.from_numpy(half_kernel).to(device)

    kernel_x = torch.cat((torch.flip(half_kernel[:, :, :, 1:], dims=(-1,)), half_kernel), dim=3)
    kernel_x = kernel_x.transpose(0, 1)
    kernel_y = kernel_x.permute(1, 0, 3, 2)
    kernel_y = torch.flipud(kernel_y)
    pads = (kernel_y.shape[-1] // 2, kernel_y.shape[-1] // 2, kernel_y.shape[-2] // 2, kernel_y.shape[-2] // 2)
    kernel_xy = fun.conv2d(fun.pad(kernel_x, pads), kernel_y, padding='same', groups=nbands)

    kernel_x = kernel_x.transpose(0, 1)
    kernel_xy = kernel_xy.transpose(0, 1)
    kernel_x = kernel_x[:, :, :, ::2]
    kernel_y = kernel_y[:, :, ::2, :]
    kernel_xy = kernel_xy[:, :, ::2, ::2]

    if direction == 'N':
        h = kernel_y
    elif direction == 'S':
        h = torch.cat((kernel_y, torch.zeros(kernel_y.shape[0], kernel_y.shape[1], 1, kernel_y.shape[3],
                                             device=device)),
                      dim=2)
    elif direction == 'W':
        h = kernel_x
    elif direction == 'E':
        h = torch.cat((kernel_x, torch.zeros(kernel_x.shape[0], kernel_x.shape[1], kernel_x.shape[2], 1,
                                             device=device)),
                      dim=3)
    elif direction == 'NW':
        h = kernel_xy
    elif direction == 'NE':
        h = torch.cat((kernel_xy, torch.zeros(kernel_xy.shape[0], kernel_xy.shape[1], kernel_xy.shape[2], 1,
                                              device=device)),
                      dim=3)
    elif direction == 'SW':
        h = torch.cat((kernel_xy, torch.zeros(kernel_xy.shape[0], kernel_xy.shape[1], 1, kernel_xy.shape[3],
                                              device=device)),
                      dim=2)
    elif direction == 'SE':
        h = torch.cat((torch.cat((kernel_xy, torch.zeros(kernel_xy.shape[0], kernel_xy.shape[1], kernel_xy.shape[2], 1,
                                                         device=device)),
                                 dim=3),
                       torch.zeros(kernel_xy.shape[0], kernel_xy.shape[1], 1, kernel_xy.shape[3] + 1,
                                   device=device)),
                      dim=2)
    else:
        h = torch.zeros(1, 1, 1, 1, device=device)  # should never happen

    shifted_img = fun.conv2d(img, h, padding='same', groups=nbands)

    return shifted_img

def half_interp23tap_kernel(nbands):
    half_kern = np.asarray([0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0,
                            0.000807762146, 0, -0.000060081482])
    half_kern = half_kern * 2.
    half_kern = np.repeat(half_kern[None, :], nbands, axis=0)
    half_kern = half_kern[:, None, None, :]
    return half_kern


def fineshift(img, shift_r, shift_c, device, sz=5):
    img = torch.clone(img).double()
    nbands = img.shape[1]
    kernel = torch.zeros(nbands, 1, sz, sz, device=device, dtype=img.dtype, requires_grad=False)

    if isinstance(shift_r, int):
        shift_r = [shift_r] * nbands
    if isinstance(shift_c, int):
        shift_c = [shift_c] * nbands
    if not torch.is_tensor(shift_r):
        shift_r = torch.tensor(shift_r, device=device, requires_grad=False)
    if not torch.is_tensor(shift_c):
        shift_c = torch.tensor(shift_c, device=device, requires_grad=False)

    r = shift_r
    c = shift_c

    r_int = r // 2
    c_int = c // 2

    r_frac = torch.remainder(r, 2)
    c_frac = torch.remainder(c, 2)

    condition = (r_frac == 1) * (c_frac == 1)
    if condition.count_nonzero() != 0:
        img[:, condition, :, :] = half_pixel_shift(img[:, condition, :, :], 'SE',
                                                   half_interp23tap_kernel(condition.count_nonzero().item()), device)
    condition = (r_frac == 1) * (c_frac != 1)
    if condition.count_nonzero() != 0:
        img[:, condition, :, :] = half_pixel_shift(img[:, condition, :, :], 'S',
                                                   half_interp23tap_kernel(condition.count_nonzero().item()), device)
    condition = (c_frac == 1) * (r_frac != 1)
    if condition.count_nonzero() != 0:
        img[:, condition, :, :] = half_pixel_shift(img[:, condition, :, :], 'E',
                                                   half_interp23tap_kernel(condition.count_nonzero().item()), device)

    cnt = sz // 2
    b = torch.tensor(range(nbands), requires_grad=False).long()
    kernel[b, :, cnt - r_int, cnt - c_int] = 1

    shifted_img = fun.conv2d(img, kernel, padding='same', groups=img.shape[1])

    return shifted_img


class DowngradeProtocol(nn.Module):
    def __init__(self, mtf, ratio, device):
        super(DowngradeProtocol, self).__init__()

        # Parameters definition
        kernel = mtf
        self.pad_size = math.floor((kernel.shape[0] - 1) / 2)
        nbands = kernel.shape[-1]
        self.ratio = ratio
        self.device = device
        # Conversion of filters in Tensor
        kernel = np.moveaxis(kernel, -1, 0)
        kernel = np.expand_dims(kernel, axis=1)

        kernel = torch.from_numpy(kernel).type(torch.float32)

        # DepthWise-Conv2d definition
        self.depthconv = nn.Conv2d(in_channels=nbands,
                                   out_channels=nbands,
                                   groups=nbands,
                                   kernel_size=kernel.shape,
                                   bias=False)

        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

        self.pad = nn.ReplicationPad2d(self.pad_size)

    def forward(self, outputs, r, c):

        x = self.pad(outputs)
        x = self.depthconv(x)
        xx = []
        for bs in range(x.shape[0]):
            xx.append(fineshift(torch.unsqueeze(x[bs], 0), r[bs], c[bs], self.device))
        x = torch.cat(xx, 0)
        if self.ratio==2:
            x = x[:, :, 1::self.ratio, 1::self.ratio]
        else:
            x = x[:, :, 2::self.ratio, 2::self.ratio]

        return x






def fspecial_gauss(size, sigma):
    """
        Function to mimic the 'fspecial' gaussian MATLAB function

        Parameters
        ----------
        size : Tuple
            The dimensions of the kernel. Dimension: H, W
        sigma : float
            The frequency of the gaussian filter

        Return
        ------
        h : Numpy array
            The Gaussian Filter of sigma frequency and size dimension

        """

    m, n = [(ss - 1.) / 2. for ss in size]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def fir_filter_wind(hd, w):
    """
        Compute fir filter with window method

        Parameters
        ----------
        hd : float
            Desired frequency response (2D)
        w : Numpy Array
            The filter kernel (2D)

        Return
        ------
        h : Numpy array
            The fir Filter

    """

    hd = np.rot90(np.fft.fftshift(np.rot90(hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = np.clip(h, a_min=0, a_max=np.max(h))
    h = h / np.sum(h)

    return h


def nyquist_filter_generator(nyquist_freq, ratio, kernel_size):
    """
        Compute the estimeted MTF filter kernels.

        Parameters
        ----------
        nyquist_freq : Numpy Array or List
            The MTF frequencies
        ratio : int
            The resolution scale which elapses between MS and PAN.
        kernel_size : int
            The size of the kernel (Only squared kernels have been implemented).

        Return
        ------
        kernel : Numpy array
            The filter based on Modulation Transfer Function.

    """

    assert isinstance(nyquist_freq, (np.ndarray, list)), 'Error: GNyq must be a list or a ndarray'

    if isinstance(nyquist_freq, list):
        nyquist_freq = np.asarray(nyquist_freq)

    nbands = nyquist_freq.shape[0]

    kernel = np.zeros((kernel_size, kernel_size, nbands))  # generic kerenel (for normalization purpose)

    fcut = 1 / np.double(ratio)

    for j in range(nbands):
        alpha = np.sqrt(((kernel_size - 1) * (fcut / 2)) ** 2 / (-2 * np.log(nyquist_freq[j])))
        H = fspecial_gauss((kernel_size, kernel_size), alpha)
        Hd = H / np.max(H)
        h = np.kaiser(kernel_size, 0.5)

        kernel[:, :, j] = np.real(fir_filter_wind(Hd, h))

    return kernel


def mtf_kernel_to_torch(h):
    """
        Compute the estimated MTF filter kernels for the supported satellites and calculate the spatial bias between
        each Multi-Spectral band and the Panchromatic (to implement the coregistration feature).

        Parameters
        ----------
        h : Numpy Array
            The filter based on Modulation Transfer Function.

        Return
        ------
        h : Tensor array
            The filter based on Modulation Transfer Function reshaped to Conv2d kernel format.
        """

    h = np.moveaxis(h, -1, 0)
    h = np.expand_dims(h, axis=1)
    h = h.astype(np.float32)
    h = torch.from_numpy(h).type(torch.float32)
    return h


def resize_images(img_ms, img_pan, ratio, sensor=None, mtf=None, apply_mtf_to_pan=False):
    """
        Function to perform a downscale of all the data provided by the satellite.
        It downsamples the data of the scale value.
        To more detail please refers to

        [1] G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. Orn Ulfarsson, L. Alparone, J. Chanussot -
            A new benchmark based on recent advances in multispectral pansharpening: Revisiting pansharpening with
            classical and emerging pansharpening methods
        [2] L. Wald, (1) T. Ranchin, (2) M. Mangolini - Fusion of satellites of different spatial resolutions:
            assessing the quality of resulting images
        [3] B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, M. Selva - MTF-tailored Multiscale Fusion of
            High-resolution MS and Pan Imagery
        [4] M. Ciotola, S. Vitale, A. Mazza, G. Poggi, G. Scarpa - Pansharpening by convolutional neural networks in
            the full resolution framework


        Parameters
        ----------
        img_ms : Numpy Array
            stack of Multi-Spectral bands. Dimension: H, W, B
        img_pan : Numpy Array
            Panchromatic Band converted in Numpy Array. Dimensions: H, W
        ratio : int
            the resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        mtf : Dictionary
            The desired Modulation Transfer Frequencies with which perform the low pass filtering process.
            Example of usage:
                MTF = {'GNyq' : np.asarray([0.21, 0.2, 0.3, 0.4]), 'GNyqPan': 0.5}
        apply_mtf_to_pan : bool
            Activate the downsample of the Panchromatic band with the Modulation Transfer Function protocol
            (Actually this feature is not used in our algorithm).


        Return
        ------
        I_MS_LR : Numpy array
            the stack of Multi-Spectral bands downgraded by the ratio factor
        I_PAN_LR : Numpy array
            The panchromatic band downsampled by the ratio factor

        """
#     def MTF(ratio, sensor, N=41):
#     if (sensor=='QB'):
#         GNyq = np.asarray([0.34, 0.32, 0.30, 0.22]) #Bands Order: B,G,R,NIR
#     elif ((sensor=='Ikonos') or (sensor=='IKONOS')):
#         GNyq = np.asarray([0.26, 0.28, 0.29, 0.28]) #Bands Order: B,G,R,NIR
#     elif (sensor=='GeoEye1') or (sensor == 'WV4'):
#         GNyq = np.asarray([0.23, 0.23, 0.23, 0.23]) #Bands Order: B, G, R, NIR
#     elif (sensor=='WV2'):
#         GNyq = 0.35 * np.ones((1, 7)); GNyq = np.append(GNyq, 0.27)
#     elif (sensor=='WV3'):
#         GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]



#     h = NyquistFilterGenerator(GNyq,ratio, N)
#     return h


# def MTF_PAN(ratio, sensor, N=41):
#     if (sensor=='QB'):
#         GNyq = np.array([0.15])
#     elif ((sensor=='Ikonos') or (sensor=='IKONOS')):
#         GNyq = np.array([0.17])
#     elif (sensor=='GeoEye1') or (sensor == 'WV4'):
#         GNyq = np.array([0.16])
#     elif (sensor=='WV2'):
#         GNyq = np.array([0.11])
#     elif (sensor=='WV3'):
#         GNyq = np.array([0.14])
#     else:
#         GNyq = np.array([0.15])
#     return NyquistFilterGenerator(GNyq, ratio, N)


    img_ms = img_ms.transpose((1,2,0))
    img_pan = img_pan.transpose((1,2,0))

    GNyq = []
    GNyqPan = []
    if (sensor is None) & (mtf is None):
        MS_scale = (math.floor(img_ms.shape[0] / ratio), math.floor(img_ms.shape[1] / ratio), img_ms.shape[2])
        PAN_scale = (math.floor(img_pan.shape[0] / ratio/2), math.floor(img_pan.shape[1] / ratio/2))
        I_MS_LR = transform.resize(img_ms, MS_scale, order=3)
        I_PAN_LR = transform.resize(img_pan, PAN_scale, order=3)
        I_MS_LR = I_MS_LR.transpose((2,0,1))
        I_PAN_LR = I_PAN_LR.transpose((2,0,1))
        return I_MS_LR, I_PAN_LR

    elif (sensor == 'QB') & (mtf is None):
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22])  # Bands Order: B,G,R,NIR
        GNyqPan = np.asarray([0.15])
    elif ((sensor == 'Ikonos') or (sensor == 'IKONOS')) & (mtf is None):
        GNyq = np.asarray([0.26, 0.28, 0.29, 0.28])  # Bands Order: B,G,R,NIR
        GNyqPan = np.asarray([0.17])
    elif (sensor == 'GeoEye1' or sensor == 'GE1') or (sensor == 'WV4') & (mtf is None):
        GNyq = np.asarray([0.23, 0.23, 0.23, 0.23])  # Bands Order: B, G, R, NIR
        GNyqPan = np.asarray([0.16])
    elif (sensor == 'WV2') & (mtf is None):
        GNyq = 0.35 * np.ones((1, 7))
        GNyq = np.append(GNyq, 0.27)
        GNyqPan = np.asarray([0.11])
    elif (sensor == 'WV3') & (mtf is None):
        GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]
        GNyqPan = np.asarray([0.14])
    elif mtf is not None:
        GNyq = mtf['GNyq']
        GNyqPan = np.asarray([mtf['GNyqPan']])

    N = 41

    b = img_ms.shape[-1]

    img_ms = np.moveaxis(img_ms, -1, 0)
    img_ms = np.expand_dims(img_ms, axis=0)

    h = nyquist_filter_generator(GNyq, ratio, N)
    h = mtf_kernel_to_torch(h)

    conv = nn.Conv2d(in_channels=b, out_channels=b, padding=math.ceil(N / 2),
                     kernel_size=h.shape, groups=b, bias=False, padding_mode='replicate')

    conv.weight.data = h
    conv.weight.requires_grad = False

    I_MS_LP = conv(torch.from_numpy(img_ms)).numpy()
    I_MS_LP = np.squeeze(I_MS_LP)
    I_MS_LP = np.moveaxis(I_MS_LP, 0, -1)
    MS_scale = (math.floor(img_ms.shape[2] / ratio), math.floor(img_ms.shape[3] / ratio), img_ms.shape[1])
    PAN_scale = (math.floor(img_pan.shape[0] / ratio/2), math.floor(img_pan.shape[1] / ratio/2))

    I_MS_LR = transform.resize(I_MS_LP, MS_scale, order=0)

    if apply_mtf_to_pan:
        img_pan = np.expand_dims(img_pan, [0, 1])

        h = nyquist_filter_generator(GNyqPan, ratio, N)
        h = mtf_kernel_to_torch(h)

        conv = nn.Conv2d(in_channels=1, out_channels=1, padding=math.ceil(N / 2),
                         kernel_size=h.shape, groups=1, bias=False, padding_mode='replicate')

        conv.weight.data = h
        conv.weight.requires_grad = False

        I_PAN_LP = conv(torch.from_numpy(img_pan)).numpy()
        I_PAN_LP = np.squeeze(I_PAN_LP)
        I_PAN_LR = transform.resize(I_PAN_LP, PAN_scale, order=0)

    else:
        I_PAN_LR = transform.resize(img_pan, PAN_scale, order=3)

    I_MS_LR = I_MS_LR.transpose((2,0,1))
    I_PAN_LR = I_PAN_LR.transpose((2,0,1))

    return I_MS_LR, I_PAN_LR


def gen_mtf(ratio, sensor, kernel_size=41):
    """
        Compute the estimated MTF filter kernels for the supported satellites.

        Parameters
        ----------
        ratio : int
            The resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        kernel_size : int
            The size of the kernel (Only squared kernels have been implemented).

        Return
        ------
        kernel : Numpy array
            The filter based on Modulation Transfer Function for the desired satellite.

        """
    nyquist_gains = []

    if sensor == 'QB':
        nyquist_gains = np.asarray([0.34, 0.32, 0.30, 0.22])  # Bands Order: B,G,R,NIR
    elif (sensor == 'Ikonos') or (sensor == 'IKONOS'):
        nyquist_gains = np.asarray([0.26, 0.28, 0.29, 0.28])  # Bands Order: B,G,R,NIR
    elif (sensor == 'GeoEye1') or (sensor == 'GE1') or (sensor == 'WV4'):
        nyquist_gains = np.asarray([0.23, 0.23, 0.23, 0.23])  # Bands Order: B, G, R, NIR
    elif sensor == 'WV2':
        nyquist_gains = 0.35 * np.ones((1, 7))
        nyquist_gains = np.append(nyquist_gains, 0.27)
    elif sensor == 'WV3':
        nyquist_gains = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]

    h = nyquist_filter_generator(nyquist_gains, ratio, kernel_size)

    return h



def interp23tap(img, ratio):
    """
        Polynomial (with 23 coefficients) interpolator Function.

        For more details please refers to:

        [1]  B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli - Context-driven fusion of high spatial and spectral
             resolution images based on oversampled multiresolution analysis
        [2] B. Aiazzi, S. Baronti, M. Selva, and L. Alparone - Bi-cubic interpolation for shift-free pan-sharpening
        [3] G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. Orn Ulfarsson, L. Alparone, J. Chanussot -
            A new benchmark based on recent advances in multispectral pansharpening: Revisiting pansharpening with
            classical and emerging pansharpening methods


        Parameters
        ----------
        img : Numpy Array
            Image to be scaled. Dimension: H, W, B
        ratio : int
            The desired scale. It must be a factor power of 2.


        Return
        ------
        img : Numpy array
            the interpolated img.

        """
    assert ((2 ** (round(math.log(ratio, 2)))) == ratio), 'Error: Only resize factors power of 2'

    r, c, b = img.shape

    CDF23 = np.asarray(
        [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0,
         -0.000060081482])
    CDF23 = [element * 2 for element in CDF23]
    BaseCoeff = np.expand_dims(np.concatenate([np.flip(CDF23[1:]), CDF23]), axis=-1)

    for z in range(int(ratio / 2)):

        I1LRU = np.zeros(((2 ** (z + 1)) * r, (2 ** (z + 1)) * c, b))

        if z == 0:
            I1LRU[1::2, 1::2, :] = img
        else:
            I1LRU[::2, ::2, :] = img

        for i in range(b):
            temp = ft.convolve(np.transpose(I1LRU[:, :, i]), BaseCoeff, mode='wrap')
            I1LRU[:, :, i] = ft.convolve(np.transpose(temp), BaseCoeff, mode='wrap')

        img = I1LRU

    return img



def upgrade(mslr,rat,device):

    b,c,r,w = mslr.shape
    out = torch.zeros(b,c,r*rat,w*rat)   
    for i in range(b):
        mslrtem = mslr[i].detach().numpy()
        mslrtem = mslrtem.transpose((1,2,0))
        ms = interp23tap_torch(mslrtem, rat, device)
        ms = ms.transpose((2,0,1))
        out[i] =torch.from_numpy(ms)
    out = out.to(device)
    return out




def interp23tap_torch(img, ratio, device):
    """
        A PyTorch implementation of the Polynomial interpolator Function.

        For more details please refers to:

        [1]  B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli - Context-driven fusion of high spatial and spectral
             resolution images based on oversampled multiresolution analysis
        [2] B. Aiazzi, S. Baronti, M. Selva, and L. Alparone - Bi-cubic interpolation for shift-free pan-sharpening
        [3] G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. Orn Ulfarsson, L. Alparone, J. Chanussot -
            A new benchmark based on recent advances in multispectral pansharpening: Revisiting pansharpening with
            classical and emerging pansharpening methods


        Parameters
        ----------
        img : Numpy Array
            Image to be scaled. The conversion in Torch Tensor is made within the function. Dimension: H, W, B
        ratio : int
            The desired scale. It must be a factor power of 2.
        device : Torch Device
            The device on which perform the operation.


        Return
        ------
        img : Numpy array
           The interpolated img.

    """
    assert ((2 ** (round(math.log(ratio, 2)))) == ratio), 'Error: Only resize factors power of 2'

    r, c, b = img.shape

    CDF23 = np.asarray(
        [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0,
         -0.000060081482])
    CDF23 = [element * 2 for element in CDF23]
    BaseCoeff = np.expand_dims(np.concatenate([np.flip(CDF23[1:]), CDF23]), axis=-1)
    BaseCoeff = np.expand_dims(BaseCoeff, axis=(0, 1))
    BaseCoeff = np.concatenate([BaseCoeff] * b, axis=0)

    BaseCoeff = torch.from_numpy(BaseCoeff).to(device)
    img = img.astype(np.float32)
    img = np.moveaxis(img, -1, 0)

    for z in range(int(ratio / 2)):

        I1LRU = np.zeros((b, (2 ** (z + 1)) * r, (2 ** (z + 1)) * c))

        if z == 0:
            I1LRU[:, 1::2, 1::2] = img
        else:
            I1LRU[:, ::2, ::2] = img

        I1LRU = np.expand_dims(I1LRU, axis=0)
        conv = nn.Conv2d(in_channels=b, out_channels=b, padding=(11, 0),
                         kernel_size=BaseCoeff.shape, groups=b, bias=False, padding_mode='circular')

        conv.weight.data = BaseCoeff
        conv.weight.requires_grad = False

        t = conv(torch.transpose(torch.from_numpy(I1LRU).to(device), 2, 3))
        img = conv(torch.transpose(t, 2, 3)).cpu().detach().numpy()
        img = np.squeeze(img)

    img = np.moveaxis(img, 0, -1)

    return img




