"""
https://github.com/matciotola/Z-PNN

"""
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import scipy.ndimage.filters as ft

from skimage.transform.integral import integral_image as integral
from math import floor, ceil


class Sensor:
    """
        Sensor class definition. It contains all the useful data for the proper management of the algorithm.

        Parameters
        ----------
        sensor : str
            The name of the sensor which has provided the image.
    """

    def __init__(self, sensor):

        self.sensor = sensor
        self.ratio = 4

        if (sensor == 'QB') or (sensor == 'GeoEye1') or (sensor == 'GE1') or (sensor == 'WV2') or (sensor == 'WV3'):
            self.kernels = [9, 5, 5]
        elif (sensor == 'Ikonos') or (sensor == 'IKONOS'):
            self.kernels = [5, 5, 5]

        if (sensor == 'QB') or (sensor == 'GeoEye1') or (sensor == 'GE1') or (sensor == 'Ikonos') or (
                sensor == 'IKONOS'):
            self.nbands = 4
        elif (sensor == 'WV2') or (sensor == 'WV3'):
            self.nbands = 8
  
        self.nbits = 11

        if sensor == 'WV2' or sensor == 'WV3':
            self.beta = 0.36
            self.learning_rate = 1e-5
        elif (sensor == 'GE1') or (sensor == 'GeoEye1'):
            self.beta = 0.25
            self.learning_rate = 5 * 1e-5


def xcorr_torch(img_1, img_2, half_width, device):
    """
        A PyTorch implementation of Cross-Correlation Field computation.

        Parameters
        ----------
        img_1 : Torch Tensor
            First image on which calculate the cross-correlation. Dimensions: 1, 1, H, W
        img_2 : Torch Tensor
            Second image on which calculate the cross-correlation. Dimensions: 1, 1, H, W
        half_width : int
            The semi-size of the window on which calculate the cross-correlation
        device : Torch Device
            The device on which perform the operation.


        Return
        ------
        L : Torch Tensor
            The cross-correlation map between img_1 and img_2

    """
    w = ceil(half_width)
    ep = 1e-20
    img_1 = img_1.type(torch.DoubleTensor)
    img_2 = img_2.type(torch.DoubleTensor)

    img_1 = img_1.to(device)
    img_2 = img_2.to(device)

    img_1 = F.pad(img_1, (w, w, w, w))
    img_2 = F.pad(img_2, (w, w, w, w))

    img_1_cum = torch.cumsum(torch.cumsum(img_1, dim=-1), dim=-2)
    img_2_cum = torch.cumsum(torch.cumsum(img_2, dim=-1), dim=-2)

    img_1_mu = (img_1_cum[:, :, 2 * w:, 2 * w:] - img_1_cum[:, :, :-2 * w, 2 * w:] - img_1_cum[:, :, 2 * w:, :-2 * w] +
                img_1_cum[:, :, :-2 * w, :-2 * w]) / (4 * w ** 2)
    img_2_mu = (img_2_cum[:, :, 2 * w:, 2 * w:] - img_2_cum[:, :, :-2 * w, 2 * w:] - img_2_cum[:, :, 2 * w:, :-2 * w] +
                img_2_cum[:, :, :-2 * w, :-2 * w]) / (4 * w ** 2)

    img_1 = img_1[:, :, w:-w, w:-w] - img_1_mu
    img_2 = img_2[:, :, w:-w, w:-w] - img_2_mu

    img_1 = F.pad(img_1, (w, w, w, w))
    img_2 = F.pad(img_2, (w, w, w, w))

    i2_cum = torch.cumsum(torch.cumsum(img_1 ** 2, dim=-1), dim=-2)
    j2_cum = torch.cumsum(torch.cumsum(img_2 ** 2, dim=-1), dim=-2)
    ij_cum = torch.cumsum(torch.cumsum(img_1 * img_2, dim=-1), dim=-2)

    sig2_ij_tot = (ij_cum[:, :, 2 * w:, 2 * w:] - ij_cum[:, :, :-2 * w, 2 * w:] - ij_cum[:, :, 2 * w:, :-2 * w] +
                   ij_cum[:, :, :-2 * w, :-2 * w])
    sig2_ii_tot = (i2_cum[:, :, 2 * w:, 2 * w:] - i2_cum[:, :, :-2 * w, 2 * w:] - i2_cum[:, :, 2 * w:, :-2 * w] +
                   i2_cum[:, :, :-2 * w, :-2 * w])
    sig2_jj_tot = (j2_cum[:, :, 2 * w:, 2 * w:] - j2_cum[:, :, :-2 * w, 2 * w:] - j2_cum[:, :, 2 * w:, :-2 * w] +
                   j2_cum[:, :, :-2 * w, :-2 * w])

    sig2_ii_tot = torch.clip(sig2_ii_tot, ep, sig2_ii_tot.max().item())
    sig2_jj_tot = torch.clip(sig2_jj_tot, ep, sig2_jj_tot.max().item())

    L = sig2_ij_tot / ((sig2_ii_tot * sig2_jj_tot) ** 0.5 + ep)

    return L


class StructuralLoss(nn.Module):

    def __init__(self, sigma, device):
        # Class initialization
        super(StructuralLoss, self).__init__()

        # Parameters definition:

        self.scale = ceil(sigma / 2)
        self.device = device

    def forward(self, outputs, labels):
        X_corr = torch.clamp(xcorr_torch(outputs, labels, self.scale, self.device), min=-1)
        X = 1.0 - X_corr

        with torch.no_grad():
            Lxcorr_no_weights = torch.mean(X)

        Lxcorr = torch.mean(X)

        return Lxcorr, Lxcorr_no_weights.item()





def cayley_dickson_property_1d(onion1, onion2):
    n = onion1.shape[1]

    if n > 1:
        middle = int(n / 2)
        a = onion1[:, :middle]
        b = onion1[:, middle:]
        neg = - torch.ones(b.shape, dtype=b.dtype, device=b.device)
        neg[:, 0] = 1
        b = b * neg
        c = onion2[:, :middle]
        d = onion2[:, middle:]
        d = d * neg

        if n == 2:
            ris = torch.cat(((a * c) - (d * b), (a * d) + (c * b)), dim=1)
        else:
            ris1 = cayley_dickson_property_1d(a, c)
            ris2 = cayley_dickson_property_1d(d, b * neg)
            ris3 = cayley_dickson_property_1d(a * neg, d)
            ris4 = cayley_dickson_property_1d(c, b)

            ris = torch.cat((ris1 - ris2, ris3 + ris4), dim=1)
    else:
        ris = onion1 * onion2

    return ris

def normalize_block(im):
    m = im.view(im.shape[0], im.shape[1], -1).mean(2).view(im.shape[0], im.shape[1], 1, 1)
    s = im.view(im.shape[0], im.shape[1], -1).std(2).view(im.shape[0], im.shape[1], 1, 1)

    s[s == 0] = 1e-10

    y = ((im - m) / s) + 1

    return y, m, s


def cayley_dickson_property_2d(onion1, onion2):
    dim3 = onion1.shape[1]
    if dim3 > 1:
        middle = int(dim3 / 2)

        a = onion1[:, 0:middle, :, :]
        b = onion1[:, middle:, :, :]
        neg = - torch.ones(b.shape, dtype=b.dtype, device=b.device)
        neg[:, 0, :, :] = 1
        b = b * neg
        c = onion2[:, 0:middle, :, :]
        d = onion2[:, middle:, :, :]

        d = d * neg
        if dim3 == 2:
            ris = torch.cat(((a * c) - (d * b), (a * d) + (c * b)), dim=1)
        else:
            ris1 = cayley_dickson_property_2d(a, c)
            ris2 = cayley_dickson_property_2d(d, b * neg)
            ris3 = cayley_dickson_property_2d(a * neg, d)
            ris4 = cayley_dickson_property_2d(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4

            ris = torch.cat((aux1, aux2), dim=1)
    else:
        ris = onion1 * onion2

    return ris


def q_index(im1, im2, size, device):
    im1 = im1.double()
    im2 = im2.double()
    neg = -torch.ones(im2.shape, dtype=im2.dtype, device=im2.device)
    neg[:, 0, :, :] = 1

    batch_size, dim3, _, _ = im1.size()

    im1, s, t = normalize_block(im1)

    condition = (s[:, 0, 0, 0] == 0)
    im2[condition] = im2[condition] - s[condition] + 1
    im2[~condition] = ((im2[~condition] - s[~condition]) / t[~condition]) + 1

    im2 = im2 * neg

    m1 = torch.mean(im1, dim=(2, 3))
    m2 = torch.mean(im2, dim=(2, 3))
    mod_q1m = torch.sqrt(torch.sum(torch.pow(m1, 2), dim=1))
    mod_q2m = torch.sqrt(torch.sum(torch.pow(m2, 2), dim=1))

    mod_q1 = torch.sqrt(torch.sum(torch.pow(im1, 2), dim=1))
    mod_q2 = torch.sqrt(torch.sum(torch.pow(im2, 2), dim=1))

    term2 = mod_q1m * mod_q2m
    term4 = torch.pow(mod_q1m, 2) + torch.pow(mod_q2m, 2)
    temp = [(size * size) / (size * size - 1)] * batch_size
    temp = torch.tensor(temp, device=device)
    int1 = torch.clone(temp)
    int2 = torch.clone(temp)
    int3 = torch.clone(temp)
    int1 = int1 * torch.mean(torch.pow(mod_q1, 2), dim=(-2, -1))
    int2 = int2 * torch.mean(torch.pow(mod_q2, 2), dim=(-2, -1))
    int3 = int3 * (torch.pow(mod_q1m, 2) + torch.pow(mod_q2m, 2))
    term3 = int1 + int2 - int3

    mean_bias = 2 * term2 / term4

    condition = (term3 == 0)
    q = torch.zeros((batch_size, dim3), device=device, dtype=mean_bias.dtype, requires_grad=False)
    q[condition, dim3 - 1] = mean_bias[condition]

    cbm = 2 / term3
    qu = cayley_dickson_property_2d(im1, im2)
    qm = cayley_dickson_property_1d(m1, m2)
    qv = (size * size) / (size * size - 1) * torch.mean(qu, dim=(-2, -1))
    q[~condition] = (qv[~condition] - (temp[:, None] * qm)[~condition])[:, :]
    q[~condition] = q[~condition] * mean_bias[~condition, None] * cbm[~condition, None]

    q[q == 0] = 1e-30

    return q

# ...
class Q2n(nn.Module):
    def __init__(self, device='cpu', q_block_size=8, q_shift=32):
        super(Q2n, self).__init__()

        self.Q_block_size = q_block_size
        self.Q_shift = q_shift
        self.device = device

    def forward(self, outputs, labels):

        bs, dim3, dim1, dim2 = labels.size()

        if math.ceil(math.log2(dim1)) - math.log2(dim1) != 0:
            difference = 2 ** (math.ceil(math.log2(dim1))) - dim1
            pads_2n = nn.ReplicationPad2d((math.floor(difference / 2), math.ceil(difference / 2), 0, 0))
            labels = pads_2n(labels)
            outputs = pads_2n(outputs)

        if math.ceil(math.log2(dim2)) - math.log2(dim2) != 0:
            difference = 2 ** (math.ceil(math.log2(dim2))) - dim2
            pads_2n = nn.ReplicationPad2d((0, 0, math.floor(difference / 2), math.ceil(difference / 2)))
            labels = pads_2n(labels)
            outputs = pads_2n(outputs)

        bs, dim3, dim1, dim2 = labels.size()

        stepx = math.ceil(dim1 / self.Q_shift)
        stepy = math.ceil(dim2 / self.Q_shift)

        if stepy <= 0:
            stepy = 1
            stepx = 1

        est1 = (stepx - 1) * self.Q_shift + self.Q_block_size - dim1
        est2 = (stepy - 1) * self.Q_shift + self.Q_block_size - dim2

        if (est1 != 0) + (est2 != 0) > 0:
            padding = torch.nn.ReflectionPad2d((0, est1, 0, est2))

            labels = padding(labels)
            outputs = padding(outputs)

        outputs = torch.round(outputs)
        labels = torch.round(labels)
        bs, dim3, dim1, dim2 = labels.size()

        if math.ceil(math.log2(dim3)) - math.log2(dim3) != 0:
            exp_difference = 2 ** (math.ceil(math.log2(dim3))) - dim3
            diff = torch.zeros((bs, exp_difference, dim1, dim2), device=self.device, requires_grad=False)
            labels = torch.cat((labels, diff), dim=1)
            outputs = torch.cat((outputs, diff), dim=1)

        values = []
        for j in range(stepx):
            values_i = []
            for i in range(stepy):
                o = q_index(labels[:, :, j * self.Q_shift:j * self.Q_shift + self.Q_block_size,
                            i * self.Q_shift: i * self.Q_shift + self.Q_block_size],
                            outputs[:, :, j * self.Q_shift:j * self.Q_shift + self.Q_block_size,
                            i * self.Q_shift: i * self.Q_shift + self.Q_block_size], self.Q_block_size,
                            self.device)
                values_i.append(o[:, :, None, None])
            values_i = torch.cat(values_i, -1)
            values.append(values_i)
        values = torch.cat(values, -2)
        q2n_index_map = torch.sqrt(torch.sum(values ** 2, dim=1))
        q2n_index = torch.mean(q2n_index_map, dim=(-2, -1))

        return q2n_index


class ReproDLambdaKhan(nn.Module):
    def __init__(self, device):
        super(ReproDLambdaKhan, self).__init__()
        self.Q2n = Q2n(device)

    def forward(self, shifted_downgraded_outputs, ms):
        q2n_index = self.Q2n(shifted_downgraded_outputs, ms)
        dlambda = 1.0 - torch.mean(q2n_index)

        return dlambda


# class SAM(nn.Module):
#     def __init__(self, reduction='mean'):
#         super(SAM, self).__init__()
#         self.reduction = reduction

#     @staticmethod
#     def forward(img1, img2):
#         if not img1.shape == img2.shape:
#             raise ValueError('Input images must have the same dimensions.')
#         # assert img1.ndim == 3 and img1.shape[2] > 1, "image n_channels should be greater than 1"
#         # img1_ = img1.astype(np.float64)
#         # img2_ = img2.astype(np.float64)
#         inner_product = (img1 * img2).sum(axis=1)
#         img1_spectral_norm = torch.sqrt((img1**2).sum(axis=1))
#         img2_spectral_norm = torch.sqrt((img2**2).sum(axis=1))
#         # numerical stability
#         cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + np.finfo(np.float64).eps)).clip(min=0, max=1)
        
#         out = torch.mean(torch.acos(cos_theta))

#         return out


class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()

    def forward(self, im_fake, im_true):
        sum1 = torch.sum(im_true * im_fake, 1)
        sum2 = torch.sum(im_true * im_true, 1)
        sum3 = torch.sum(im_fake * im_fake, 1)
        t = (sum2 * sum3) ** 0.5
        numlocal = torch.gt(t, 0)
        num = torch.sum(numlocal)
        t = sum1 / t
        angle = torch.acos(t)
        sumangle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle).sum()
        if num == 0:
            averangle = sumangle
        else:
            averangle = sumangle / num
        SAM = averangle * 180 / 3.14159256
        return SAM




# https://github.com/CalvinYang0/CRNet/blob/master/util/util.py
    
from math import exp
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel, device):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1).to(device)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size)
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        # self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        window = create_window(self.window_size, channel, img1.device)
        self.window = window
        self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()

    window = create_window(window_size, channel, img1.device)
    return _ssim(img1, img2, window, window_size, channel, size_average)


class ERGAS(nn.Module):
    def __init__(self, ratio=4, reduction='mean'):
        super(ERGAS, self).__init__()
        self.ratio = ratio
        self.reduction = reduction

    def forward(self, outputs, labels):
        mu = torch.mean(labels, dim=(2, 3)) ** 2
        nbands = labels.size(dim=1)
        error = torch.mean((outputs - labels) ** 2, dim=(2, 3))
        ergas_index = 100 / self.ratio * torch.sqrt(torch.sum(error / mu, dim=1) / nbands)
        if self.reduction == 'mean':
            ergas = torch.mean(ergas_index)
        else:
            ergas = torch.sum(ergas_index)

        return ergas







def sid(s1, s2):
    """
    Computes the spectral information divergence between two vectors.

    Parameters:
        s1: `numpy array`
            The first vector.

        s2: `numpy array`
            The second vector.

    Returns: `float`
            Spectral information divergence between s1 and s2.

    Reference
        C.-I. Chang, "An Information-Theoretic Approach to SpectralVariability,
        Similarity, and Discrimination for Hyperspectral Image"
        IEEE TRANSACTIONS ON INFORMATION THEORY, VOL. 46, NO. 5, AUGUST 2000.

    """
    p = (s1 / np.sum(s1)) + np.spacing(1)
    q = (s2 / np.sum(s2)) + np.spacing(1)
    out = np.nansum(p * np.log(p / q) + q * np.log(q / p))
    return out

class SID(torch.nn.Module):
    """
    Computes the spectral information divergence between two vectors.

    Parameters:
        s1: `numpy array`
            The first vector.

        s2: `numpy array`
            The second vector.

    Returns: `float`
            Spectral information divergence between s1 and s2.

    Reference
        C.-I. Chang, "An Information-Theoretic Approach to SpectralVariability,
        Similarity, and Discrimination for Hyperspectral Image"
        IEEE TRANSACTIONS ON INFORMATION THEORY, VOL. 46, NO. 5, AUGUST 2000.

    """
    def __init__(self):
        super(SID, self).__init__()
       
    def forward(self, s1, s2):
       
        finfo = torch.finfo(torch.float32)
        fin = finfo.eps

        p = (s1 / torch.sum(s1)) +fin
        q = (s2 / torch.sum(s2)) +fin

        return torch.sum(p * torch.log(p / q) + q * torch.log(q / p))
    
import scipy.io as io         
def main():

    # img = torch.randn( 4, 256, 256)
    # pan = torch.randn( 4, 256, 256)
    gtdir = r"D:\codes\pansharpening-DLPan-Toolbox-main\02-Test-toolbox-for-traditional-and-DL(Matlab)\1_TestData\WV4\W4_Mexi_Urb_W4_RR.mat"
    predir = r"D:\codes\pansharpening-DLPan-Toolbox-main\02-Test-toolbox-for-traditional-and-DL(Matlab)\2_DL_Result\WV4\mypnn\output_W4_Mexi_Urb_W4_RR+.mat"

    gti = io.loadmat(gtdir)
    prei = io.loadmat(predir)

    gt = gti['I_GT']
    pre = prei['mypnn_MS']
    x1 = sid(gt,pre)

    gt = torch.tensor(gt.astype(np.float32))
    pre = torch.tensor(pre.astype(np.float32))
    FSAM = SID()
    x = FSAM(gt,pre)


    print('.................')

if __name__ == "__main__":
    main()