#!/bin/bash

#----------------------------------------------------------------------#
#                            INITIALIZATION                            #
#----------------------------------------------------------------------#
# This is the configuration we used for training our best DeepViewAgg
# model on ScanNet, with results as stated in our paper
# https://arxiv.org/abs/2204.07548

# Select you GPU
I_GPU=0

DATA_ROOT="/mnt/hdd/datasets/"                        # set your dataset root directory, where the data was/will be downloaded
EXP_NAME="baseline_new"                                # whatever suits your needs
TASK="segmentation"
MODELS_CONFIG="${TASK}/multimodal/adl4cv-scannet"                         # family of multimodal models using the sparseconv3d backbone
MODEL_NAME="Res16UNet34-L4-intermediate-ade20k-interpolate-concat-fusion"          # specific model name
DATASET_CONFIG="${TASK}/multimodal/scannet-sparse"
TRAINING="scannet_benchmark/minkowski"             # training configuration for discriminative learning rate on the model
EPOCHS=150
BATCH_SIZE=1                                                            # 4 fits in a 32G V100. Can be increased at inference time, of course
WORKERS=4                                                               # adapt to your machine
BASE_LR=0.1                                                             # initial learning rate
LR_SCHEDULER='constant'                                              # learning rate scheduler for 60 epochs
EVAL_FREQUENCY=5                                                       # frequency at which metrics will be computed on Val. The less the faster the training but the less points on your validation curves
SUBMISSION=False                                                        # True if you want to generate files for a submission to the ScanNet 3D semantic segmentation benchmark
CHECKPOINT_DIR=''                                                       # optional path to an already-existing checkpoint. If provided, the training will resume where it was left

export SPARSE_BACKEND=torchsparse
# export SPARSE_BACKEND=minkowski

#----------------------------------------------------------------------#
#                                 RUN                                  #
#----------------------------------------------------------------------#

python -W ignore train.py \
data=${DATASET_CONFIG} \
models=${MODELS_CONFIG} \
model_name=${MODEL_NAME} \
task=${TASK} \
training=${TRAINING} \
lr_scheduler=${LR_SCHEDULER} \
eval_frequency=${EVAL_FREQUENCY} \
data.dataroot=${DATA_ROOT} \
training.cuda=${I_GPU} \
training.batch_size=${BATCH_SIZE} \
training.epochs=${EPOCHS} \
training.num_workers=${WORKERS} \
training.optim.base_lr=${BASE_LR} \
training.wandb.log=True \
training.wandb.name=${EXP_NAME} \
tracker_options.make_submission=${SUBMISSION} \
training.checkpoint_dir=${CHECKPOINT_DIR}
