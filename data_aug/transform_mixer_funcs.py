import torch
import numpy as np
import math
 
torch.pi = math.pi

#Stochastic interpolant

class SI:
    def __init__(self):
        pass
    def xt(self, t, x0, x1):
        t = torch.tensor(t)
        z = torch.randn(len(x0))
        return self.I(t,x0,x1)+self.gamma(t)*z
    def I(self, t, x0, x1):
        pass
    def gamma(self,t):
        pass
    
class trigonometric_interpolant(SI):
    def I(self, t, x0, x1):
        return self.a(t)*x0+self.b(t)*x1
    def a(self,t):
        return torch.sqrt(1-torch.pow(self.gamma(t),2))*torch.cos(torch.pi*t/2)
    def b(self,t):
        return torch.sqrt(1-torch.pow(self.gamma(t),2))*torch.sin(torch.pi*t/2)
    
    def gamma(self,t):
        return torch.sqrt(2*t*(1-t))

class trigonometric_interpolant_no_noise(SI):
    def I(self, t, x0, x1):
        return self.a(t)*x0+self.b(t)*x1
    def a(self,t):
        return torch.sqrt(1-torch.pow(self.gamma(t),2))*torch.cos(torch.pi*t/2)
    def b(self,t):
        return torch.sqrt(1-torch.pow(self.gamma(t),2))*torch.sin(torch.pi*t/2)
    def gamma(self,t):
        return torch.tensor(0)
    
class linear_interpolant(SI):
    def I(self, t, x0, x1):
        return self.a(t)*x0+self.b(t)*x1
    def a(self,t):
        return 1-t 
    def b(self,t):
        return t
    def gamma(self,t):
        return torch.sqrt(2*t*(1-t))
    
class Gaussian_enc_dec(SI):
    def I(self, t, x0, x1):
        return self.a(t)*x0+self.b(t)*x1
    def a(self,t):
        if(t>=0 and t<0.5):
            return torch.pow(torch.cos(torch.pi*t),2)
        else:
            return 0 
    def b(self,t):
        if(t>0.5 and t<=1):
            return torch.pow(torch.cos(torch.pi*t),2)
        else:
            return 0 
    def gamma(self,t):
        return torch.pow(torch.sin(torch.pi*t),2)

class Sigmoid(SI):
    def I(self, t, x0, x1):
        return self.a(t)*x0+self.b(t)*x1
    def a(self,t):
        c1 = torch.tensor(10)
        return torch.exp(-c1*(t-0.5))/(1+torch.exp(-c1*(t-0.5)))
    def b(self,t):
        c1 = torch.tensor(10)
        return 1/(1+torch.exp(-c1*(t-0.5)))
    def gamma(self,t):
        return torch.tensor(0)

    
#radial mixer
#guney
def radial_mixer(fft_img, mix_amp, mix_phs, ampcoef=0, phscoef=0, 
                 ampbw=[[0.0, 1.0]], phsbw=[[0.0, 1.0]], true_center=False):

    fft_abs = torch.abs(fft_img)

    # index of the img_center
    if true_center:
        c0, c1 = (fft_abs.shape[1]-1)/2, (fft_abs.shape[2]-1)/2
    else:
        c0, c1 = fft_abs.shape[1]//2, fft_abs.shape[2]//2

    radius = np.hypot(fft_abs.shape[1], fft_abs.shape[2]) / 2 # the radius of the circle bounding the image square
    # all the possible np.sqrt(x**2+y**2)/radius for all possible index x and index y of the image
    # shape: fft_abs[1], fft_abs[2] 
    radial_frequencies = np.sqrt(np.add.outer((np.arange(fft_abs.shape[1]) - c0)**2,   
                                              (np.arange(fft_abs.shape[2]) - c1)**2)) / radius
    # reviewing the mask: returning a matrix of the same size with entries T/F according to the conditional statement
    # only mixing the specific ranges(intervals) of the frequencies
    # note that the mask is of the same size of the fft_abs
    mask_amp = torch.zeros_like(fft_abs)
    for interval in ampbw:
        mask_amp += torch.tensor(((radial_frequencies >= interval[0]) & (radial_frequencies <= interval[1])).astype(np.float32)).to(fft_abs.device)
        
    mask_phs = torch.zeros_like(fft_abs)
    for interval in phsbw:
        mask_phs += torch.tensor(((radial_frequencies >= interval[0]) & (radial_frequencies <= interval[1])).astype(np.float32)).to(fft_abs.device)
    
    mask_amp, mask_phs = (mask_amp > 0).to(fft_abs.dtype), (mask_phs > 0).to(fft_abs.dtype) # set variable type the same as fft_abs

    # The mixing of amplitudes: using abs(acutually is the amp)**2 with mix_amp**2
    # The mixing of phases: using phases with mix_phs
    fft_abs = torch.sqrt((fft_abs ** 2) * (1 - ampcoef * mask_amp) + (mix_amp ** 2) * (ampcoef * mask_amp))
    fft_phs = torch.exp(1j*((1 - phscoef * mask_phs) * torch.angle(fft_img) + phscoef * mix_phs * mask_phs))
    fft_img = fft_abs * fft_phs

    return fft_img
