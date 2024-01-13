# Introduction
The codes of the semester project Signal Processing Techniques for Data Augmentation in Medical Image Analysis are shown here.
We use a SimCLR architecture to evaluate some Fourier-based transforms (augmentation methods).

Each experiment based on the SimCLR method is composed of pertaining and a linear evaluation. On Cifar-10,
we train for 1000 epochs and use a mini-batch size 256. ResNet-18 architecture is used as the base encoder network. We train for 500 epochs for the linear evaluation without any data augmentation. For more experimental settings, please check the report.
# Installation
    conda env create --name simclr --file env.yml  
    conda activate simclr


# File Instruction
Python files 
* simclr.py: The training process for the encoder.
* run.py: The main program that performs encoder training with method A.
* runB.py: The main program that performs encoder training with method B.
* utils.py: Some utility functions. 


Folders
* data_aug: Various designed classes for data augmentation.
* exceptions: Exception Error file.
* models: model of the encoder (ResNet).
* feature_eval: Linear evaluation for the pre-trained encoder.

# How to implement the experiments?
## Pre-train the encoder
* Create a `datasets` folder under the main path. 
* Run the shell script to pre-train the encoder according to the argument settings (There are two example shell scripts, `runA.sh` and `runB.sh`).
## Linear evaluation
* When the training is finished, copy the configuration file (e.g., config.yml) and the encoder checkpoint (e.g., exp_resnet18_amp_gm_v1_b_256_e_1000_ckp.pth.tar) to the `feature_eval/folder` folder.
* Run the shell script to do the linear evaluation (There is an example shell script, `run_eval.sh`).

