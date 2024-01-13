"""
A class that generates an augmented dataset. 

Init parameters:

root_folder: Data folder.
aug_type: Augmentation method name.
args: Arguments passed in.

Functions:

get_our_transform: Get the pipeline transforms applied to the dataset.
standard_transform: The standard SimCLR transform.
get_dataset: Apply the data augentation on the dataset and get the augmented dataset.
ContrastiveLearningViewGenerator: Generating views for each image of the dataset.
"""

from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from data_aug.data_aug import *


cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2023, 0.1994, 0.2010]


class ContrastiveLearningDataset:
    def __init__(self, root_folder, aug_type, **kwargs):
        self.root_folder = root_folder
        self.aug_type = aug_type
        self.args = kwargs['args']
        
    def get_our_transform(self, size, s=1):  
        if(self.aug_type=="default"):
            s=1
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            data_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * size)),
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="none"):
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="noneB"):
            data_transforms = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="amp_gm_v1"):
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                gaussian_mix_random_v1(),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="amp_gm_v1B"):
            data_transforms = transforms.Compose([
                # transforms.ToTensor(),
                gaussian_mix_random_v1(),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="amp_gm_v1_control"):
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                gaussian_mix_random_v1_control(self.args.max_coeff_amp),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="amp_gm_v1_controlB"):
            data_transforms = transforms.Compose([
                # transforms.ToTensor(),
                gaussian_mix_random_v1_control(self.args.max_coeff_amp),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="amp_gm_v5"):
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                gaussian_mix_random_v5(),
                transforms.Normalize(cifar10_mean, cifar10_std)])
        elif(self.aug_type=="amp_GED"):
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                amp_GED(),
                transforms.Normalize(cifar10_mean, cifar10_std)])
        elif(self.aug_type=="phs_sig"):
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                phs_Sigmoid(),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="phs_sigB"):
            data_transforms = transforms.Compose([
                # transforms.ToTensor(),
                phs_Sigmoid(),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="phs_sig_control"):
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                phs_Sigmoid_control(self.args.max_coeff_phs),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="phs_TI_square"):
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                phs_TI_square(),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="phs_TI_squareB"):
            data_transforms = transforms.Compose([
                # transforms.ToTensor(),
                phs_TI_square(),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="phs_TI_square_control"):
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                phs_TI_square_control(self.args.max_coeff_phs),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="phs_TI_no_noise_control"):
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                phs_TI_no_noise_control(self.args.max_coeff_phs),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="phs_hsv_TI_no_noise_control"):
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                phs_hsv_TI_no_noise_new_control(self.args.max_coeff_phs),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="phs_TI_square_controlB"):
            data_transforms = transforms.Compose([
                # transforms.ToTensor(),
                phs_TI_square_control(self.args.max_coeff_phs),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="amp_gm_v1_phs_sig"):
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                gaussian_mix_random_v1(),
                phs_Sigmoid(),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="amp_gm_v1_phs_TI_square"):
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                gaussian_mix_random_v1(),
                phs_TI_square(),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="amp_gm_v1_phs_TI_squareB"):
            data_transforms = transforms.Compose([
                # transforms.ToTensor(),
                gaussian_mix_random_v1(),
                phs_TI_square(),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="amp_gm_v1_phs_TI_square_control"):
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                gaussian_mix_random_v1_control(self.args.max_coeff_amp),
                phs_TI_square_control(self.args.max_coeff_phs),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="amp_gm_v1_phs_TI_square_controlB"):
            data_transforms = transforms.Compose([
                # transforms.ToTensor(),
                gaussian_mix_random_v1_control(self.args.max_coeff_amp),
                phs_TI_square_control(self.args.max_coeff_phs),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="amp_gm_v4_phs_TI_square"):
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                gaussian_mix_random_v4(),
                phs_TI_square(),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        elif(self.aug_type=="amp_GED_phs_TI_square"):
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                amp_GED(),
                phs_TI_square(),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ])
        return data_transforms
    
    def standard_transform(self, size, s=1):
        # simclr standard transform s
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor(),
            ])
        return data_transforms
    
    def get_dataset(self, name, n_views, p):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_our_transform(32),
                                                                  n_views, 
                                                                  p, 
                                                                  variantB=self.args.variantB,
                                                                  standard_transform=self.standard_transform(32)
                                                                  ),
                                                              download=False)
                                                              }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
