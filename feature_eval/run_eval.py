"""
Linear evaluation (training the last fc layer).
"""
import argparse
import torch 
import sys
import os
import yaml
import torchvision
import wandb
import logging

from utils_eval import *


parser = argparse.ArgumentParser(description='PyTorch SimCLR evaluation')

parser.add_argument('--eval-epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--eval-batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--eval-download', default=False, type=bool,
                    help='download the dataset')
parser.add_argument('--projectname', type=str)
parser.add_argument('--folder', type=str, help='folder containing the configuration file and the checkpoint')

def main():

    args = parser.parse_args()


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    with open(os.path.join('./'+args.folder+'/config.yml')) as file:
        config = yaml.load(file, Loader=yaml.UnsafeLoader)

    set_seed(config.seed)

    if config.arch == 'resnet18':
        model = torchvision.models.resnet18(num_classes=10).to(device)
    elif config.arch == 'resnet50':
        model = torchvision.models.resnet50(num_classes=10).to(device)
    print("encoder archietecture:", config.arch)

    # load pretrained encoder
    ckp_file_name = "exp_"+config.arch+"_"+config.aug_type+"_b_"+f"{config.batch_size}"+"_e_"+f"{config.epochs}"+'_ckp.pth.tar'
    checkpoint = torch.load('./'+args.folder+'/'+ckp_file_name, map_location=device)
    state_dict = checkpoint['state_dict']   
    print("Load checkpoint file:", ckp_file_name)
    # print(state_dict.keys())

    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]
    # print(state_dict.keys())

    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']


    # load datasets
    data_download = args.eval_download
    if config.dataset_name == 'cifar10':
        train_loader, test_loader = get_cifar10_data_loaders(download=data_download, batch_size=args.eval_batch_size)
    elif config.dataset_name == 'stl10':
        train_loader, test_loader = get_stl10_data_loaders(download=data_download, batch_size=args.eval_batch_size)
    print("Dataset:", config.dataset_name)

    # init wandb 
    run_name = wandb_run_name(config,args)
    run = wandb.init(
    project=args.projectname, 
    name=run_name,
    )
    logging.basicConfig(filename=os.path.join(run.dir, 'training.log'), level=logging.DEBUG)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    # train classifier and evaluate
    logging.info(f"Start linear classifer training for {args.eval_epochs} epochs.")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(device)


    for epoch in range(args.eval_epochs):
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        top1_train_accuracy /= (counter + 1)
        top1_accuracy = 0
        top5_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
        
            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]
        
        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        print(f"Epoch {epoch+1}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
        if epoch % 1 == 0:
            wandb.log({'train accuracy':top1_train_accuracy.item()}, step=epoch+1)
            wandb.log({'top1 test accuracy':top1_accuracy.item()}, step=epoch+1)
            wandb.log({'top5 test accuracy':top5_accuracy.item()}, step=epoch+1)
        
        logging.debug(f"Epoch: {epoch+1}\tTrain accuracy: {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test accuracy: {top5_accuracy.item()}")

    logging.info("Training has finished.")
    # save config
    save_config_file(run.dir, args) # save eval config
    shutil.copy('./'+args.folder+'/config.yml', run.dir) # copy simclr training config
    logging.info(f"config file is saved.")

    # save model checkpoints
    checkpoint_name = ckp_file_name[:-12]+'_eval_ckp.pth.tar'
    save_checkpoint({
        'epoch': args.eval_epochs,
        'arch': config.arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=False, filename=os.path.join(run.dir, checkpoint_name))
    logging.info(f"Model checkpoint and metadata has been saved at {run.dir}.")

    #finish wandb
    wandb.finish()

if __name__ == "__main__":
    main()  
