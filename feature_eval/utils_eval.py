import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import yaml
import os
import shutil
import random
import numpy as np


def get_stl10_data_loaders(download, shuffle=False, batch_size=256):
  
  data_path = './data'

  train_dataset = datasets.STL10(data_path, split='train', download=download,
                                  transform=transforms.ToTensor())

  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
  
  test_dataset = datasets.STL10(data_path, split='test', download=download,
                                  transform=transforms.ToTensor())

  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=3, drop_last=False, shuffle=shuffle)
  return train_loader, test_loader

def get_cifar10_data_loaders(download, shuffle=False, batch_size=256):
  
  data_path = './data'

  cifar10_mean = [0.4914, 0.4822, 0.4465]
  cifar10_std = [0.2023, 0.1994, 0.2010]

  train_dataset = datasets.CIFAR10(data_path, train=True, download=download,
                                  transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ]))

  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
  
  test_dataset = datasets.CIFAR10(data_path, train=False, download=download,
                                  transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std)
                ]))

  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=3, drop_last=False, shuffle=shuffle)
  return train_loader, test_loader

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config_eval.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)
        
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def wandb_run_name(config,args):
    if(config.n_views==2):
        run_name = "exp_"+config.arch+"_"+config.aug_type+"_b_"+f"{config.batch_size}"+'_'+f"{config.use_simclr}"+'_'+f"{config.use_fftclr}"+'_p'+f"{config.prob_transform}"
    if(config.n_views==4):
        run_name = "exp_"+config.arch+"_"+config.aug_type+"_b_"+f"{config.batch_size}"+'_p'+f"{config.prob_transform}"
    if(config.variantB == 1):
        run_name+='variant'
    if(config.aug_type == 'phs_TI_square_control' or config.aug_type == 'phs_TI_square_controlB' \
       or config.aug_type =='phs_sig_control' or config.aug_type =='phs_TI_no_noise_control'):
        run_name+='_maxcoeffphs'+f'{config.max_coeff_phs}'
    if(config.aug_type == 'amp_gm_v1_control' or config.aug_type == 'amp_gm_v1_controlB'):
        run_name+='_maxcoeffamp'+f'{config.max_coeff_amp}'
    if(config.aug_type =='amp_gm_v1_phs_TI_square_control'):
        run_name+='_maxcoeffamp'+f'{config.max_coeff_amp}'+'_maxcoeffphs'+f'{config.max_coeff_phs}'
    
    run_name = run_name+'_eval_'+f"{args.eval_epochs}"
    
    return run_name


def set_seed(seed):

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True