"""Interpolant classes."""
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

    
