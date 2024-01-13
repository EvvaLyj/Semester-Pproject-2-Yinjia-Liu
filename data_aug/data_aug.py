"""Custom transform classes."""

from data_aug.transform_funcs import *
import torch
import numpy as np

class gaussian_mix(object):
    def __init__(self, coeff_amp):
        super(gaussian_mix, self).__init__()
        self.coeff_amp = coeff_amp
    def __call__(self, img_tensor):
        results = Gaussian_mix_up(img_tensor,  mode1="Gaussian_mix", mode2='a', coeff_amp=self.coeff_amp)
        img_tensor_result = results['output'].float()
        return img_tensor_result 

class gaussian_mix_random_v1(object):
    def __init__(self):
        super(gaussian_mix_random_v1, self).__init__()
    def __call__(self, img_tensor):
        self.coeff_amp = np.random.uniform()  # random coeff_amp in [0,1]
        results = Gaussian_mix_up(img_tensor,  mode1="Gaussian_mix", mode2='a', gma_v=1, coeff_amp=self.coeff_amp)
        img_tensor_result = results['output'].float()
        return img_tensor_result 
    def __repr__(self):
        return self.__class__.__name__+'()'

class gaussian_mix_random_v4(object):
    def __init__(self):
        super(gaussian_mix_random_v4, self).__init__()
    def __call__(self, img_tensor):
        self.coeff_amp = np.random.uniform()  # random coeff_amp in [0,1]
        results = Gaussian_mix_up(img_tensor,  mode1="Gaussian_mix", mode2='a', gma_v=4, coeff_amp=self.coeff_amp)
        img_tensor_result = results['output'].float()
        return img_tensor_result 
    def __repr__(self):
        return self.__class__.__name__+'()'

class gaussian_mix_random_v5(object):
    def __init__(self):
        super(gaussian_mix_random_v5, self).__init__()
    def __call__(self, img_tensor):
        self.coeff_amp = np.random.uniform()  # random coeff_amp in [0,1]
        results = Gaussian_mix_up(img_tensor,  mode1="Gaussian_mix", mode2='a', gma_v=5, coeff_amp=self.coeff_amp)
        img_tensor_result = results['output'].float()
        return img_tensor_result 
    def __repr__(self):
        return self.__class__.__name__+'()'

class phs_Sigmoid(object):
    def __init__(self):
        super(phs_Sigmoid, self).__init__()
    def __call__(self, img_tensor):
        self.coeff_phs = np.random.uniform()  # random coeff_amp in [0,1]
        results = stochastic_interpolant(img_tensor,'Sig','p',self.coeff_phs)
        img_tensor_result = results['output'].float()
        return img_tensor_result 
    def __repr__(self):
        return self.__class__.__name__+'()'
    
class phs_TI_square(object):
    def __init__(self):
        super(phs_TI_square, self).__init__()
    def __call__(self, img_tensor):
        self.coeff_phs = np.random.uniform()  # random coeff_phs in [0,1]
        results = stochastic_interpolant(img_tensor,'TI_square','p',self.coeff_phs)
        img_tensor_result = results['output'].float()
        return img_tensor_result 
    def __repr__(self):
        return self.__class__.__name__+'()'


class amp_GED(object):
    def __init__(self):
        super(amp_GED, self).__init__()
    def __call__(self, img_tensor):
        self.coeff_amp = np.random.uniform()  # random coeff_amp in [0,1]
        results = stochastic_interpolant(img_tensor,'GED','a',self.coeff_amp)
        img_tensor_result = results['output'].float()
        return img_tensor_result 
    def __repr__(self):
        return self.__class__.__name__+'()'

class phs_Sigmoid_control(object):
    def __init__(self, max_coeff_phs):
        super(phs_Sigmoid_control, self).__init__()
        self.max_coeff_phs = max_coeff_phs
    def __call__(self, img_tensor):
        self.coeff_phs = np.random.uniform(0,self.max_coeff_phs)  # random coeff_amp in [0,1]
        results = stochastic_interpolant(img_tensor,'Sig','p',self.coeff_phs)
        img_tensor_result = results['output'].float()
        return img_tensor_result 
    def __repr__(self):
        return self.__class__.__name__+'()'
    
class phs_TI_square_control(object):
    def __init__(self, max_coeff_phs):
        super(phs_TI_square_control, self).__init__()
        self.max_coeff_phs = max_coeff_phs
    def __call__(self, img_tensor):
        self.coeff_phs = np.random.uniform(0,self.max_coeff_phs)  # random coeff_phs in [0,max_coeff_phs]
        results = stochastic_interpolant(img_tensor,'TI_square','p',self.coeff_phs)
        img_tensor_result = results['output'].float()
        return img_tensor_result 
    def __repr__(self):
        return self.__class__.__name__+'()'


class phs_TI_no_noise_control(object):
    def __init__(self, max_coeff_phs):
        super(phs_TI_no_noise_control, self).__init__()
        self.max_coeff_phs = max_coeff_phs
    def __call__(self, img_tensor):
        self.coeff_phs = np.random.uniform(0,self.max_coeff_phs)  # random coeff_phs in [0,1]
        results = stochastic_interpolant(img_tensor,'TI_no_noise','p',self.coeff_phs)
        img_tensor_result = results['output'].float()
        return img_tensor_result 
    def __repr__(self):
        return self.__class__.__name__+'()'
    
class gaussian_mix_random_v1_control(object):
    def __init__(self, max_coeff_amp):
        super(gaussian_mix_random_v1_control, self).__init__()
        self.max_coeff_amp = max_coeff_amp
    def __call__(self, img_tensor):
        self.coeff_amp = np.random.uniform(0,self.max_coeff_amp)  # random coeff_amp in [0,max_coeff_amp]
        results = Gaussian_mix_up(img_tensor,  mode1="Gaussian_mix", mode2='a', gma_v=1, coeff_amp=self.coeff_amp)
        img_tensor_result = results['output'].float()
        return img_tensor_result 
    def __repr__(self):
        return self.__class__.__name__+'()'

class phs_hsv_TI_no_noise_new_control(object):
    def __init__(self, max_coeff_phs):
        super(phs_hsv_TI_no_noise_new_control, self).__init__()
        self.max_coeff_phs = max_coeff_phs
    def __call__(self, img_tensor):
        self.coeff_phs = np.random.uniform(0,self.max_coeff_phs)  # random coeff_phs in [0,1]
        results = stochastic_interpolant_new(img_tensor,'TI_no_noise','p_hsv',self.coeff_phs, select_indx=[False, False, True])
        img_tensor_result = results['output'].float()
        return img_tensor_result 
    def __repr__(self):
        return self.__class__.__name__+'()'