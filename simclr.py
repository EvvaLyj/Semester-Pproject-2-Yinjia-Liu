import logging
import os
import sys
import wandb

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint, wandb_run_name, set_seed

# torch.manual_seed(0)

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        # wandb init
        run_name = wandb_run_name(self.args)
        self.run = wandb.init(
            project=self.args.projectname, 
            name=run_name, 
            )
        
        logging.basicConfig(filename=os.path.join(self.run.dir, 'training.log'), level=logging.DEBUG)
        
    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.run.dir, self.args)
        logging.info(f"config file is saved.")

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    wandb.log({'loss':loss}, step=epoch_counter+1)
                    wandb.log({'top1 accuracy':top1[0]}, step=epoch_counter+1)
                    wandb.log({'top5 accuracy':top5[0]}, step=epoch_counter+1)
                    wandb.log({'learning rate':self.scheduler.get_lr()[0]}, step=epoch_counter+1)
                n_iter += 1

            # save model checkpoints
            if((epoch_counter+1)%200==0):
                checkpoint_name = "exp_"+self.args.arch+"_"+self.args.aug_type+"_b_"+f"{self.args.batch_size}"+"_e_"+f"{epoch_counter+1}"+'_ckp.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter+1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.run.dir, checkpoint_name))
                logging.info(f"Epoch{epoch_counter+1}.Model checkpoint and metadata has been saved at {self.run.dir}.")

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter+1}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")


        wandb.finish()

    def train_fft(self, train_loader_simclr, train_loader_fftclr):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.run.dir, self.args)
        logging.info(f"config file is saved.")

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):

            dataloader_iterator_simclr = iter(train_loader_simclr)
            for images_fft, _ in tqdm(train_loader_fftclr):
                try:
                    images_simclr,_ = next(dataloader_iterator_simclr)
                except StopIteration:
                    dataloader_iterator_simclr = iter(train_loader_simclr)
                    images_simclr,_ = next(dataloader_iterator_simclr)
                
                
                images_fft = torch.cat(images_fft,dim=0)
                images_fft = images_fft.to(self.args.device)
                # logging.debug(f'Debug:{torch.isnan(images_fft).any()}')

                images_simclr = torch.cat(images_simclr,dim=0)
                images_simclr = images_simclr.to(self.args.device)
                # logging.debug(f'Debug:{torch.isnan(images_simclr).any()}')

                with autocast(enabled=self.args.fp16_precision):
                    loss =  torch.tensor(0.).to(self.args.device)
                    
                    features_fft = self.model(images_fft)
                    logits_fft, labels_fft = self.info_nce_loss(features_fft)
                    loss_fft = self.criterion(logits_fft, labels_fft)

                    features_simclr = self.model(images_simclr)
                    logits_simclr, labels_simclr = self.info_nce_loss(features_simclr)
                    loss_simclr = self.criterion(logits_simclr, labels_simclr)

                    if(self.args.use_fftclr):
                        loss += loss_fft
                    if(self.args.use_simclr):
                        loss += loss_simclr

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    wandb.log({'loss':loss}, step=epoch_counter+1)
                    wandb.log({'loss_simclr':loss_simclr}, step=epoch_counter+1)
                    wandb.log({'loss_fft':loss_fft}, step=epoch_counter+1)
                    wandb.log({'learning rate':self.scheduler.get_lr()[0]}, step=epoch_counter+1)
                
                n_iter += 1

            # save model checkpoints
            if((epoch_counter+1)%200==0):
                checkpoint_name = "exp_"+self.args.arch+"_"+self.args.aug_type+"_b_"+f"{self.args.batch_size}"+"_e_"+f"{epoch_counter+1}"+'_ckp.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter+1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.run.dir, checkpoint_name))
                logging.info(f"Epoch{epoch_counter+1}.Model checkpoint and metadata has been saved at {self.run.dir}.")

            # warmup 
            if epoch_counter >= self.args.warmup_epoch:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter+1}\tLoss: {loss}")

        logging.info("Training has finished.")


        wandb.finish()

    def train_fftB(self, train_loader):
        #combination B: 4 views for one image

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.run.dir, self.args)
        logging.info(f"config file is saved.")

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    wandb.log({'loss':loss}, step=epoch_counter+1)
                    wandb.log({'learning rate':self.scheduler.get_lr()[0]}, step=epoch_counter+1)
                n_iter += 1

            # save model checkpoints
            if((epoch_counter+1)%200==0):
                checkpoint_name = "exp_"+self.args.arch+"_"+self.args.aug_type+"_b_"+f"{self.args.batch_size}"+"_e_"+f"{epoch_counter+1}"+'_ckp.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter+1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.run.dir, checkpoint_name))
                logging.info(f"Epoch{epoch_counter+1}.Model checkpoint and metadata has been saved at {self.run.dir}.")

            # warmup
            if epoch_counter >= self.args.warmup_epoch:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter+1}\tLoss: {loss}")

        logging.info("Training has finished.")


        wandb.finish()


    def train_fft_load(self, train_loader_simclr, train_loader_fftclr, checkpoint):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.run.dir, self.args)
        logging.info(f"config file is saved.")

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        epochs_needed = self.args.epochs - epoch
        logging.info(f"Load checkpoint at epoch {epoch}. Still need training for {epochs_needed} epochs.")

        for epoch_counter in range(epoch, self.args.epochs):

            dataloader_iterator_simclr = iter(train_loader_simclr)
            for images_fft, _ in tqdm(train_loader_fftclr):
                try:
                    images_simclr,_ = next(dataloader_iterator_simclr)
                except StopIteration:
                    dataloader_iterator_simclr = iter(train_loader_simclr)
                    images_simclr,_ = next(dataloader_iterator_simclr)
                
                
                images_fft = torch.cat(images_fft,dim=0)
                images_fft = images_fft.to(self.args.device)
                # logging.debug(f'Debug:{torch.isnan(images_fft).any()}')

                images_simclr = torch.cat(images_simclr,dim=0)
                images_simclr = images_simclr.to(self.args.device)
                # logging.debug(f'Debug:{torch.isnan(images_simclr).any()}')

                with autocast(enabled=self.args.fp16_precision):
                    loss =  torch.tensor(0.).to(self.args.device)
                    
                    features_fft = self.model(images_fft)
                    logits_fft, labels_fft = self.info_nce_loss(features_fft)
                    loss_fft = self.criterion(logits_fft, labels_fft)

                    features_simclr = self.model(images_simclr)
                    logits_simclr, labels_simclr = self.info_nce_loss(features_simclr)
                    loss_simclr = self.criterion(logits_simclr, labels_simclr)

                    if(self.args.use_fftclr):
                        loss += loss_fft
                    if(self.args.use_simclr):
                        loss += loss_simclr

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    wandb.log({'loss':loss}, step=epoch_counter+1)
                    wandb.log({'loss_simclr':loss_simclr}, step=epoch_counter+1)
                    wandb.log({'loss_fft':loss_fft}, step=epoch_counter+1)
                    wandb.log({'learning rate':self.scheduler.get_lr()[0]}, step=epoch_counter+1)
                
                n_iter += 1

            # save model checkpoints
            if((epoch_counter+1)%200==0):
                checkpoint_name = "exp_"+self.args.arch+"_"+self.args.aug_type+"_b_"+f"{self.args.batch_size}"+"_e_"+f"{epoch_counter+1}"+'_ckp.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter+1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.run.dir, checkpoint_name))
                logging.info(f"Epoch{epoch_counter+1}.Model checkpoint and metadata has been saved at {self.run.dir}.")

            # # warmup 
            if epoch_counter >= self.args.warmup_epoch:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter+1}\tLoss: {loss}")

        logging.info("Training has finished.")


        wandb.finish()

    def train_fftB_load(self, train_loader, checkpoint):
        #combination B: 4 views for one image

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.run.dir, self.args)
        logging.info(f"config file is saved.")

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        epochs_needed = self.args.epochs - epoch
        logging.info(f"Load checkpoint at epoch {epoch}. Still need training for {epochs_needed} epochs.")

        for epoch_counter in range(epoch, self.args.epochs):
            for images, _ in tqdm(train_loader):
                
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    wandb.log({'loss':loss}, step=epoch_counter+1)
                    wandb.log({'learning rate':self.scheduler.get_lr()[0]}, step=epoch_counter+1)
                n_iter += 1

            # save model checkpoints
            if((epoch_counter+1)%200==0):
                checkpoint_name = "exp_"+self.args.arch+"_"+self.args.aug_type+"_b_"+f"{self.args.batch_size}"+"_e_"+f"{epoch_counter+1}"+'_ckp.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter+1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.run.dir, checkpoint_name))
                logging.info(f"Epoch{epoch_counter+1}.Model checkpoint and metadata has been saved at {self.run.dir}.")

            # # warmup
            if epoch_counter >= self.args.warmup_epoch:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter+1}\tLoss: {loss}")

        logging.info("Training has finished.")


        wandb.finish()
