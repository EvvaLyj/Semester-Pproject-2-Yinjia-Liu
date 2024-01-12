import os
import shutil
import numpy as np
import random
import torch
import yaml
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)


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

def wandb_run_name(args):
    if(args.n_views==2):
        run_name = "exp_"+args.arch+"_"+args.aug_type+"_b_"+f"{args.batch_size}"+'_'+f"{args.use_simclr}"+'_'+f"{args.use_fftclr}"+'_p'+f"{args.prob_transform}"
    if(args.n_views==4):
        run_name = "exp_"+args.arch+"_"+args.aug_type+"_b_"+f"{args.batch_size}"+'_p'+f"{args.prob_transform}"
    if(args.variantB == 1):
        run_name+='variant'
    if(args.aug_type == 'phs_TI_square_control' or args.aug_type == 'phs_sig_control' or \
       args.aug_type == 'phs_TI_square_controlB' or args.aug_type == 'phs_TI_no_noise_control' or \
        args.aug_type == 'phs_hsv_TI_no_noise_control'):
        run_name+='_maxcoeffphs'+f'{args.max_coeff_phs}'
    if(args.aug_type == 'amp_gm_v1_control' or args.aug_type == 'amp_gm_v1_controlB'):
        run_name+='_maxcoeffamp'+f'{args.max_coeff_amp}'
    if(args.aug_type == 'amp_gm_v1_phs_TI_square_control' or args.aug_type == 'amp_gm_v1_phs_TI_square_controlB'):
        run_name+='_maxcoeffamp'+f'{args.max_coeff_amp}'+'_maxcoeffphs'+f'{args.max_coeff_phs}'
    return run_name


def dataset_train_fft(args):
    #prepare datasets for train_fft
    dataset_simclr = ContrastiveLearningDataset(args.data, "default", args=args)
    train_dataset_simclr = dataset_simclr.get_dataset(args.dataset_name, args.n_views, args.prob_transform)
    train_loader_simclr = torch.utils.data.DataLoader(
    train_dataset_simclr, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, drop_last=True)
    
    dataset_fftclr = ContrastiveLearningDataset(args.data, args.aug_type, args=args)
    train_dataset_fftclr = dataset_fftclr.get_dataset(args.dataset_name, args.n_views, args.prob_transform)
    train_loader_fftclr = torch.utils.data.DataLoader(
    train_dataset_fftclr, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, drop_last=True)

    return train_loader_simclr, train_loader_fftclr


def dataset_train_fftB(args):
    #prepare datasets for train_fft
    dataset = ContrastiveLearningDataset(args.data, args.aug_type, args=args)
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views, args.prob_transform)
    train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, drop_last=True)

    return train_loader

def set_seed(seed):

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
