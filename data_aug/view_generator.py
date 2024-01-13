"""
Generating views for each image of the dataset. 

our_transform: Our Fourier-based transform.
n_views： Take n_views random views of one image.
p: The probability of applying the transforms on the view.

"""

import numpy as np
from torchvision.transforms import transforms


cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2023, 0.1994, 0.2010]

class ContrastiveLearningViewGenerator(object):
    
    def __init__(self, our_transform, n_views=2, p=0.5, **args):
        self.our_transform = our_transform 
        self.n_views = n_views
        self.p = p
        
        if(args['variantB']==1):
            self.p2 = 0
        else:
            self.p2 = p
        
        self.standard_transform = args['standard_transform'] # simclr standard transform s
        self.base_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std)
            ])
        self.base_transformB = transforms.Compose([
                transforms.Normalize(cifar10_mean, cifar10_std)
            ])
    
    def __call__(self, x):
        if(self.n_views!=2 and self.n_views!=4):
            return [self.our_transform(x) for i in range(self.n_views)]

        # Combine A
        elif(self.n_views==2):
            x_transformed = [self.our_transform(x)]
            # Apply our_transform for the second view with a probability p
            if(random_unit(self.p)):
                x_transformed.append(self.our_transform(x))
            else:
                x_transformed.append(self.base_transform(x))
            return x_transformed
        
        # Combine B
        elif(self.n_views==4):
            if(np.isclose(self.p,1)):
                x_transform_2 = [self.standard_transform(x) for i in range(2)]
                x_transform_4 = [self.our_transform(x) for x in x_transform_2 for i in range(2)]
            else:
                x_transform_A = self.standard_transform(x)
                x_transform_B = self.standard_transform(x)
                x_transform_4 = []
                x_transform_4.append(self.our_transform(x_transform_A))
                x_transform_4.append(self.our_transform(x_transform_B))

                # Apply our_transform for the two second views in the second augmentation layer with a probability p and p2
                if(random_unit(self.p)):
                    x_transform_4.append(self.our_transform(x_transform_A))
                else:
                    x_transform_4.append(self.base_transformB(x_transform_A))

                if(random_unit(self.p2)):
                    x_transform_4.append(self.our_transform(x_transform_B))
                else:
                    x_transform_4.append(self.base_transformB(x_transform_B))

            return x_transform_4
        
def random_unit(p: float):
    R = np.random.uniform(0,1)
    return R<p


