import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
from utils import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-aug', '--aug-type', default='noneB', type=str,
                    help='data augmentation type', choices=['noneB',
                                                            'amp_gm_v1B', 'amp_gm_v1_controlB',
                                                            'phs_TI_squareB','phs_TI_square_controlB',
                                                            'amp_gm_v1_phs_TI_squareB','amp_gm_v1_phs_TI_square_controlB'
                                                            ])
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 3)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

parser.add_argument('--warmup-epoch', default=10, type=int)
parser.add_argument('--prob-transform', default=0.5, type=float, help='the probability of applying the transforms on the second view.')
parser.add_argument('--use-simclr', default=0, type=int)
parser.add_argument('--use-fftclr', default=1, type=int)
parser.add_argument('--projectname', type=str)

# if use aug_type xx_control, adjust the following hyperparameters
parser.add_argument('--max-coeff-phs', type=float, default=0.5)
parser.add_argument('--max-coeff-amp', type=float, default=0.5)

parser.add_argument('--variantB', help='If 1, use a variant architecture of B method.',
                    type=int, default=0)
parser.add_argument('--ckp-path', type=str)

def main():
    args = parser.parse_args()
    # assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available

    # seed
    if args.seed is not None:
        set_seed(args.seed)
        print(f"You have chosen to seed training. The seed is {args.seed}.")

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    train_loader = dataset_train_fftB(args)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)
    ckp = torch.load(args.ckp_path)
    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train_fftB_load(train_loader,ckp)

if __name__ == "__main__":
    main()


