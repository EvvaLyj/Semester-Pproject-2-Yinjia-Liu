import torch
import numpy as np
from torch import nn
from data_aug.transform_mixer_funcs import *


def FFT_img(image, interp="bilinear"):
    # Resize the image such that the size is the power of 2
    # Return the magtitude spectrum and phase spectrum 

    # input size: #channels * height * width

    image = image.to(torch.float32)
    # image_resized, original_size = resize_to_power_of_2(image, mode=interp)
    # print(f'execute resizing.\n original size:{original_size}, new size: {image_resized.shape[-2:]}')
    image_resized = image
    original_size = image.shape[-2:]

    fft_img_amp = torch.zeros_like(image_resized)
    fft_img_phs = torch.zeros_like(image_resized)
    for c in range(image_resized.shape[0]):

        fft_img =  torch.fft.fftshift(torch.fft.fftn(image_resized[c]))
        fft_img_amp[c] = torch.abs(fft_img)
        fft_img_phs[c] = torch.angle(fft_img)
        
    return fft_img_amp, fft_img_phs
def IFFT_img(fft_img_amp, fft_img_phs, original_size, interp='bilinear'):
    # original size: height * weight 
    # fft_img_amp: #channels * fft_height * fft_weight
    ifft_img = torch.zeros_like(fft_img_amp)
    for c in range(fft_img_amp.shape[0]):
        fft = fft_img_amp[c] * torch.exp(1j*fft_img_phs[c])
        ifft_img[c] = torch.fft.ifftn(torch.fft.ifftshift(fft)).real
        # ifft_img[c] = np.abs(torch.fft.ifftn(torch.fft.ifftshift(fft)))
    ifft_img = torch.nn.functional.interpolate(ifft_img.unsqueeze(0), 
        size=original_size, mode=interp, align_corners=True).squeeze(0)   
    return ifft_img


def gaussian_kernel(size=5, sigma=1.):
    """\
    creates gaussian kernel with side length `size` and a sigma of `sigma`
    """
    """
    # https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def freq_process(img_tensor, interp='bilinear', mode1='Gaussian_mix', mode2='a', gma_v=0, g_sigma=5, **args):

    vmax = torch.max(img_tensor)
    vmin = torch.min(img_tensor)

    results = {}
    n_channel = img_tensor.shape[0]
    image_tensor = img_tensor.float()#image tensor --> FloatTensor 

    # FFT
    fft_image_amp, fft_image_phs = FFT_img(image_tensor);results.update({'fft_amp':fft_image_amp,'fft_phs':fft_image_phs})
    
    if(mode1 == 'Gaussian_mix'):

        if(mode2 =='p'):
            
            if not np.isclose(args['coeff_phs'],0):
                g_img_phs = torch.tensor(gaussian_kernel(size=image_tensor.shape[-1], sigma=g_sigma))
                # g_img_phs = g_img_phs*torch.sum(fft_image_phs) 
                g_img_phs = g_img_phs.repeat(n_channel,1,1) # its dim aligns with orignal img
                results.update({'g_phs':g_img_phs})
                fft_img_mix_phs = (1-args['coeff_phs'])*fft_image_phs+ args['coeff_phs']*g_img_phs
            else:
                fft_img_mix_phs = fft_image_phs
        
            results.update({'fft_mix_g_phs':fft_img_mix_phs})
            ifft_img_mix_phs = IFFT_img(fft_image_amp, fft_img_mix_phs, image_tensor.shape[-2:])
            results.update({'output':ifft_img_mix_phs})
        
        
        elif(mode2 =='a'):
            
            # version 0: mix directly on the amp 
            if (gma_v ==0): 
                if not np.isclose(args['coeff_amp'],0):
                    g_img_amp = torch.tensor(gaussian_kernel(size=image_tensor.shape[-1], sigma=g_sigma))
                    g_img_amp = g_img_amp.repeat(n_channel,1,1) # its dim aligns with orignal img
                    results.update({'g_amp':g_img_amp})
                    fft_img_mix_amp = (1-args['coeff_amp'])*fft_image_amp + args['coeff_amp']*g_img_amp
                else: 
                    fft_img_mix_amp = fft_image_amp

                results.update({'fft_mix_g_amp':fft_img_mix_amp})
                ifft_img_mix_amp = IFFT_img(fft_img_mix_amp, fft_image_phs, image_tensor.shape[-2:])
                results.update({'output':ifft_img_mix_amp})
            # version 1: mix directly on the amp, and keep the gaussian image the same energy as the amp 
            if (gma_v ==1): 
                if not np.isclose(args['coeff_amp'],0):
                    g_img_amp = torch.tensor(gaussian_kernel(size=image_tensor.shape[-1], sigma=g_sigma))
                    g_img_energy = torch.sum(g_img_amp**2)
                    fft_img_energy = torch.sum(fft_image_amp**2)
                    g_img_amp = g_img_amp*torch.sqrt(fft_img_energy/(n_channel*g_img_energy)) #--------------------------------------add this line
                    g_img_amp = g_img_amp.repeat(n_channel,1,1) # its dim aligns with orignal img
                    results.update({'g_amp':g_img_amp})
                    fft_img_mix_amp = (1-args['coeff_amp'])*fft_image_amp + args['coeff_amp']*g_img_amp
                else: 
                    fft_img_mix_amp = fft_image_amp

                results.update({'fft_mix_g_amp':fft_img_mix_amp})
                ifft_img_mix_amp = IFFT_img(fft_img_mix_amp, fft_image_phs, image_tensor.shape[-2:])
                results.update({'output':ifft_img_mix_amp})
            # version 2: mix on the log amp 
            if (gma_v ==2): 
                if not np.isclose(args['coeff_amp'],0):
                    g_img_amp = torch.tensor(gaussian_kernel(size=image_tensor.shape[-1], sigma=g_sigma))
                    g_img_amp = g_img_amp.repeat(n_channel,1,1) # its dim aligns with orignal img
                    results.update({'g_amp':g_img_amp})
                    fft_img_mix_amp = torch.exp((1-args['coeff_amp'])*torch.log(fft_image_amp) + args['coeff_amp']*g_img_amp)# -------the change is here
                else: 
                    fft_img_mix_amp = fft_image_amp

                results.update({'fft_mix_g_amp':fft_img_mix_amp})
                ifft_img_mix_amp = IFFT_img(fft_img_mix_amp, fft_image_phs, image_tensor.shape[-2:])
                results.update({'output':ifft_img_mix_amp})
            # version 3: version 0-> parseval
            if (gma_v ==3): 
                if not np.isclose(args['coeff_amp'],0):
                    g_img_amp = torch.tensor(gaussian_kernel(size=image_tensor.shape[-1], sigma=g_sigma))
                    g_img_amp = g_img_amp.repeat(n_channel,1,1) # its dim aligns with orignal img
                    results.update({'g_amp':g_img_amp})
                    fft_img_amp_squared, g_img_amp_squared = fft_image_amp**2, g_img_amp**2
                    fft_img_mix_amp =  torch.sqrt((1-args['coeff_amp'])*fft_img_amp_squared + args['coeff_amp']*g_img_amp_squared)
                else: 
                    fft_img_mix_amp = fft_image_amp

                results.update({'fft_mix_g_amp':fft_img_mix_amp})
                ifft_img_mix_amp = IFFT_img(fft_img_mix_amp, fft_image_phs, image_tensor.shape[-2:])
                results.update({'output':ifft_img_mix_amp})
            # version 4: version 1-> parseval
            if (gma_v ==4): 
                if not np.isclose(args['coeff_amp'],0):
                    g_img_amp = torch.tensor(gaussian_kernel(size=image_tensor.shape[-1], sigma=g_sigma))
                    g_img_energy = torch.sum(g_img_amp**2)
                    fft_img_energy = torch.sum(fft_image_amp**2)
                    g_img_amp = g_img_amp*torch.sqrt(fft_img_energy/g_img_energy) 
                    g_img_amp = g_img_amp.repeat(n_channel,1,1) 
                    results.update({'g_amp':g_img_amp})
                    fft_img_amp_squared, g_img_amp_squared = fft_image_amp**2, g_img_amp**2
                    fft_img_mix_amp =  torch.sqrt((1-args['coeff_amp'])*fft_img_amp_squared + args['coeff_amp']*g_img_amp_squared)
                else: 
                    fft_img_mix_amp = fft_image_amp

                results.update({'fft_mix_g_amp':fft_img_mix_amp})
                ifft_img_mix_amp = IFFT_img(fft_img_mix_amp, fft_image_phs, image_tensor.shape[-2:])
                results.update({'output':ifft_img_mix_amp})
            # version 5: version 1-> norm and denorm 
            if (gma_v ==5): 
                if not np.isclose(args['coeff_amp'],0):
                    # norm
                    fft_image_amp_mean = torch.mean(fft_image_amp)
                    fft_image_amp_std = torch.std(fft_image_amp)
                    fft_image_amp_norm = (fft_image_amp-fft_image_amp_mean)/fft_image_amp_std
                    # mix with gaussian image
                    g_img_amp = torch.tensor(gaussian_kernel(size=image_tensor.shape[-1], sigma=g_sigma))
                    g_img_energy = torch.sum(g_img_amp**2)
                    fft_img_energy = torch.sum(fft_image_amp_norm**2)
                    g_img_amp = g_img_amp*torch.sqrt(fft_img_energy/g_img_energy) 
                    g_img_amp = g_img_amp.repeat(n_channel,1,1) 
                    results.update({'g_amp':g_img_amp})
                    fft_img_mix_amp = (1-args['coeff_amp'])*fft_image_amp_norm + args['coeff_amp']*g_img_amp
                    # denorm
                    fft_img_mix_amp = fft_img_mix_amp*fft_image_amp_std + fft_image_amp_mean
                else: 
                    fft_img_mix_amp = fft_image_amp

                results.update({'fft_mix_g_amp':fft_img_mix_amp})
                ifft_img_mix_amp = IFFT_img(fft_img_mix_amp, fft_image_phs, image_tensor.shape[-2:])
                results.update({'output':ifft_img_mix_amp})


    results_output_bound = bound_result(vmin, vmax, results['output'])
    results.update({'output':results_output_bound})
    return results
   
def gau_mix_difference(img_tensor,coeff_amp,norm=False):
    # g(t)-g(1)
    results_t = freq_process(img_tensor, norm=norm, mode1="Gaussian_mix", mode2='a', coeff_amp=coeff_amp)
    results_1 = freq_process(img_tensor, norm=norm, mode1="Gaussian_mix", mode2='a', coeff_amp=1)

    results = results_t["output"]-results_1["output"]

    return results

# all-channel mean
def stochastic_interpolant(img_tensor, mode, mode2, t):
    vmax = torch.max(img_tensor)
    vmin = torch.min(img_tensor)

    results = {}
    n_channel = img_tensor.shape[0]
    image_tensor = img_tensor.float()#image tensor --> FloatTensor 
    
    # FFT: fft_image_amp, fft_image_phs have a shape of (n_channels, height, width)
    fft_image_amp, fft_image_phs = FFT_img(image_tensor);results.update({'fft_amp':fft_image_amp,'fft_phs':fft_image_phs})
    
    if mode2=='a':
    # amp interpolation
        if not np.isclose(t,0):
            # compute x0 and x1 + normalization 
            x0 = fft_image_amp.float().flatten()
            x0_mean = torch.mean(x0);x0_std = torch.std(x0)
                        
            x0 = (x0-x0_mean)/x0_std
            x1 = torch.ones_like(x0)*torch.mean(x0)
            xt = torch.zeros_like(x0)
            #interpolation
            if(mode=='LI'):
                interpolant = linear_interpolant()
                xt = interpolant.xt(t, x0, x1)
            if(mode=='GED'):
                interpolant = Gaussian_enc_dec()
                xt = interpolant.xt(t, x0, x1)
            if(mode=='TI'):
                interpolant = trigonometric_interpolant()
                xt = interpolant.xt(t, x0, x1)
            if(mode=='TI_square'):
                interpolant = trigonometric_interpolant()
                xt = interpolant.xt(t**2, x0, x1)
            # denorm
            xt = xt*x0_std+x0_mean
            #reshape + abs
            fft_img_mix_amp = torch.abs(xt.reshape(fft_image_amp.shape))
            
        else: 
            fft_img_mix_amp = fft_image_amp

        results.update({'fft_mix_amp':fft_img_mix_amp})
        ifft_img_mix_amp = IFFT_img(fft_img_mix_amp, fft_image_phs, image_tensor.shape[-2:])
        results.update({'output':ifft_img_mix_amp})

    elif mode2=='p':
    # phs interpolation
        if not np.isclose(t,0) and torch.is_nonzero(torch.std(fft_image_phs)): # in case that the phs is totally zero (flat) leading to 0 std and NAN phs value
            # compute x0 and x1 + normalization 
            x0 = fft_image_phs.float().flatten()
            x0_mean = torch.mean(x0);x0_std = torch.std(x0)

            x0_ori = x0
            x0 = (x0-x0_mean)/x0_std
            x1 = torch.ones_like(x0)*torch.mean(x0)
            xt = torch.zeros_like(x0)
            #interpolation
            if(mode=='LI'):
                interpolant = linear_interpolant()
                xt = interpolant.xt(t, x0, x1)
            if(mode=='GED'):
                interpolant = Gaussian_enc_dec()
                xt = interpolant.xt(t, x0, x1)
            if(mode=='TI'):
                interpolant = trigonometric_interpolant()
                xt = interpolant.xt(t, x0, x1)
            if(mode=='TI_square'):
                interpolant = trigonometric_interpolant()
                xt = interpolant.xt(t**2, x0, x1)
            if(mode=='TI_square_no_noise'):
                interpolant = trigonometric_interpolant_no_noise()
                xt = interpolant.xt(t**2, x0, x1)
            if(mode=='TI_no_noise'):
                interpolant = trigonometric_interpolant_no_noise()
                xt = interpolant.xt(t, x0, x1)
            # denorm
            xt = xt*x0_std+x0_mean
           
            # Sigmoid does not need normalization
            if(mode=='Sig'):
                interpolant = Sigmoid()
                xt = interpolant.xt(t, x0_ori, torch.mean(x0_ori)*torch.ones_like(x0_ori))
                
            #reshape 
            fft_img_mix_phs = xt.reshape(fft_image_phs.shape)      
        else: 
            fft_img_mix_phs = fft_image_phs

        results.update({'fft_mix_phs':fft_img_mix_phs})
        ifft_img_mix_phs = IFFT_img(fft_image_amp, fft_img_mix_phs, image_tensor.shape[-2:])
        results.update({'output':ifft_img_mix_phs})

    results_output_bound = bound_result(vmin, vmax, results['output'])
    results.update({'output':results_output_bound})
    return results

# 3-channel mean
def stochastic_interpolant_new(img_tensor, mode, mode2, t, **args):

    vmax = torch.max(img_tensor)
    vmin = torch.min(img_tensor)

    results = {}
    n_channel = img_tensor.shape[0]
    image_tensor = img_tensor.float()#image tensor --> FloatTensor 
    
    # FFT: fft_image_amp, fft_image_phs have a shape of (n_channels, height, width)
    fft_image_amp, fft_image_phs = FFT_img(image_tensor);results.update({'fft_amp':fft_image_amp,'fft_phs':fft_image_phs})
    
    if mode2=='a':
        fft_img_mix_amp = torch.zeros_like(fft_image_amp)
    elif mode2 =='p':
        fft_img_mix_phs = torch.zeros_like(fft_image_phs)
    elif mode2 =='p_hsv':
        convertor = RGB_HSV()
        fft_image_phs_norm = (fft_image_phs+torch.pi)/(2*torch.pi)
        fft_image_phs_hsv = convertor.rgb_to_hsv(fft_image_phs_norm.unsqueeze(0)) # (B * 3 * H * W)
        results.update({'fft_image_phs_hsv':fft_image_phs_hsv})
        result_phs_hsv = fft_image_phs_hsv.clone() 

    # interpolation for each channel
    for c in range(img_tensor.shape[0]): # channels
        
        if not np.isclose(t,0):
            # compute x0 and x1 + normalization 
            if mode2=='a':
                x0 = fft_image_amp[c].float().flatten()
            elif mode2 =='p':
                x0 = fft_image_phs[c].float().flatten()
            elif mode2 =='p_hsv' and not args["select_indx"][c]:
                continue
            elif mode2 =='p_hsv':
                x0 = fft_image_phs_hsv[0,c].float().flatten()
            x0_mean = torch.mean(x0);x0_std = torch.std(x0)

            if(x0_std==0):
                # if x0 has 0 std, do not interpolate (else there is NAN)! (very likely x0 is all-zero)
                if mode2=='a':
                    fft_img_mix_amp[c] = fft_image_amp[c]
                    continue
                elif mode2 =='p':
                    fft_img_mix_phs[c] = fft_image_phs[c]
                    continue
            
            x0_ori = x0
            x0 = (x0-x0_mean)/x0_std
            x1 = torch.ones_like(x0)*torch.mean(x0)
            xt = torch.zeros_like(x0)
            #interpolation
            if(mode=='LI'):
                interpolant = linear_interpolant()
                xt = interpolant.xt(t, x0, x1)
            if(mode=='GED'):
                interpolant = Gaussian_enc_dec()
                xt = interpolant.xt(t, x0, x1)
            if(mode=='TI'):
                interpolant = trigonometric_interpolant()
                xt = interpolant.xt(t, x0, x1)
            if(mode=='TI_square'):
                interpolant = trigonometric_interpolant()
                xt = interpolant.xt(t**2, x0, x1)
            if(mode=='TI_square_no_noise'):
                interpolant = trigonometric_interpolant_no_noise()
                xt = interpolant.xt(t**2, x0, x1)
            if(mode=='TI_no_noise'):
                interpolant = trigonometric_interpolant_no_noise()
                xt = interpolant.xt(t, x0, x1)
            # denorm
            xt = xt*x0_std+x0_mean

            # Sigmoid does not need normalization
            if(mode=='Sig'):
                interpolant = Sigmoid()
                xt = interpolant.xt(t, x0_ori, torch.mean(x0_ori)*torch.ones_like(x0_ori))

            #reshape + abs for amp, reshape for phs/phs_hsv
            if mode2=='a':
                fft_img_mix_amp[c] = torch.abs(xt.reshape(fft_image_amp[c].shape))
            elif  mode2 == 'p':
                fft_img_mix_phs[c] = xt.reshape(fft_image_phs[c].shape)
            elif  mode2 == 'p_hsv':
                result_phs_hsv[0,c] = xt.reshape(fft_image_phs_hsv[0,c].shape)
        
        else: 
            if mode2=='a':
                fft_img_mix_amp[c] = fft_image_amp[c]
            elif  mode2 == 'p':
                fft_img_mix_phs[c] = fft_image_phs[c]

    # IFFT  
    if mode2 == 'a':
        results.update({'fft_mix_amp':fft_img_mix_amp})
        ifft_img_mix_amp = IFFT_img(fft_img_mix_amp, fft_image_phs, image_tensor.shape[-2:])
        results.update({'output':ifft_img_mix_amp})

    elif mode2 == 'p':
        results.update({'fft_mix_phs':fft_img_mix_phs})
        ifft_img_mix_phs = IFFT_img(fft_image_amp, fft_img_mix_phs, image_tensor.shape[-2:])
        results.update({'output':ifft_img_mix_phs})
    elif mode2 =="p_hsv":
        #hsv->rgb
        result_phs_hsv = bound_hsv_result(result_phs_hsv)
        results.update({'result_phs_hsv':result_phs_hsv})
        fft_img_mix_phs = convertor.hsv_to_rgb(result_phs_hsv)
        fft_img_mix_phs = fft_img_mix_phs[0]*(2*torch.pi)-torch.pi
        results.update({'fft_mix_phs':fft_img_mix_phs})
        #ifft
        ifft_img_mix_phs = IFFT_img(fft_image_amp, fft_img_mix_phs, image_tensor.shape[-2:])
        results.update({'output':ifft_img_mix_phs})

    results_output_bound = bound_result(vmin, vmax, results['output'])
    results.update({'output':results_output_bound})
    return results

def bound_result(img_min,img_max, xt):
    # clip
    out = torch.min(img_max,xt)
    out = torch.max(img_min,out)

    return out

def bound_hsv_result(result_phs_hsv):
    #h [0,1] periodicity
    #s [0,1]
    #v [0,1]

    result_phs_hsv[0,0] = result_phs_hsv[0,0]%1
    result_phs_hsv[0,1]=torch.max(torch.zeros_like(result_phs_hsv[0,1]),result_phs_hsv[0,1])
    result_phs_hsv[0,1]=torch.min(torch.ones_like(result_phs_hsv[0,1]),result_phs_hsv[0,1])
    result_phs_hsv[0,2]=torch.max(torch.zeros_like(result_phs_hsv[0,2]),result_phs_hsv[0,2])
    result_phs_hsv[0,2]=torch.min(torch.ones_like(result_phs_hsv[0,2]),result_phs_hsv[0,2])

    return result_phs_hsv

# https://blog.csdn.net/Brikie/article/details/115086835
"""
Pytorch implementation of RGB convert to HSV, and HSV convert to RGB,
RGB or HSV's shape: (B * C * H * W)
RGB or HSV's range: [0, 1)
"""


class RGB_HSV(nn.Module):
    def __init__(self, eps=1e-8):
        super(RGB_HSV, self).__init__()
        self.eps = eps

    def rgb_to_hsv(self, img):

        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,2]==img.max(1)[0] ]
        hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,1]==img.max(1)[0] ]
        hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

        hue[img.min(1)[0]==img.max(1)[0]] = 0.0
        hue = hue/6

        saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + self.eps )
        saturation[ img.max(1)[0]==0 ] = 0

        value = img.max(1)[0]
        
        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value],dim=1)
        return hsv

    def hsv_to_rgb(self, hsv):
        h,s,v = hsv[:,0,:,:],hsv[:,1,:,:],hsv[:,2,:,:]
        #对出界值的处理
        h = h%1
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
  
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))
        
        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5
        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
        
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb
