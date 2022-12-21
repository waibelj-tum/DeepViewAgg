
import os
import sys
import torch
from time import time
from omegaconf import OmegaConf

import warnings
warnings.filterwarnings('ignore')

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer import Trainer
from torch_points3d.utils.config import hydra_read




@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))

    trainer = Trainer(cfg)
    model = trainer._model

    if not train: # Just loads in one batch and does forward pass to visualize
        train_loader = trainer._dataset.train_dataloader

        batch = next(iter(train_loader))
        model.set_input(batch, model.device)
        batch = model(batch) # Forward pass

        out = model.output # Output stored in model attributes

        #torchviz.make_dot(out, params=dict(list(model.named_parameters()))).render("torchviz", format="png")
    else:
        trainer.train()

        GlobalHydra.get_state().clear()
        print("Done!")


if __name__ == "__main__":
    # Select you GPU
    I_GPU = 0
    torch.cuda.set_device(I_GPU)
    DIR = '/home/rozenberszki/altay/DeepViewAgg'
    ROOT = os.path.join(DIR, "..")
    sys.path.insert(0, ROOT)
    sys.path.insert(0, DIR) 

    torch.cuda.empty_cache()
    train = True

    DATA_ROOT="/mnt/hdd/datasets"                                       # set your dataset root directory, where the data was/will be downloaded
    EXP_NAME="dev-debug"                              # whatever suits your needs
    TASK="segmentation"
    MODELS_CONFIG=f"{TASK}/multimodal/adl4cv"                         # family of multimodal models using the sparseconv3d backbone
    MODEL_NAME="base-early-local-fusion"      # specific model name
    DATASET_CONFIG=f"{TASK}/multimodal/kitti360-sparse"
    TRAINING="kitti360_benchmark/sparseconv3d"     # training configuration for discriminative learning rate on the model
    EPOCHS=60
    CYLINDERS_PER_EPOCH=12000                                               # roughly speaking, 40 cylinders per window
    TRAINVAL=False                                                          # True to train on Train+Val (eg before submission)
    MINI=True                                                               # True to train on mini version of KITTI-360 (eg to debug)
    BATCH_SIZE=1                                                            # 4 fits in a 32G V100. Can be increased at inference time, of course
    WORKERS=4                                                               # adapt to your machine
    BASE_LR=0.02                                                             # initial learning rate
    LR_SCHEDULER='constant'                                                 # learning rate scheduler for 60 epochs
    EVAL_FREQUENCY=1                                                        # frequency at which metrics will be computed on Val. The less the faster the training but the less points on your validation curves
    SUBMISSION=False                                                        # True if you want to generate files for a submission to the KITTI-360 3D semantic segmentation benchmark
    CHECKPOINT_DIR=""                                                       # optional path to an already-existing checkpoint. If provided, the training will resume where it was left

    overrides = [
        'task=segmentation',
        f'data={DATASET_CONFIG}',
        f'models={MODELS_CONFIG}',
        f'model_name={MODEL_NAME}',
        f'task={TASK}',
        f'training={TRAINING}',
        f'lr_scheduler={LR_SCHEDULER}',
        f'eval_frequency={EVAL_FREQUENCY}',
        f'data.sample_per_epoch={CYLINDERS_PER_EPOCH}',
        f'data.dataroot={DATA_ROOT}',
        f'data.train_is_trainval={TRAINVAL}',
        f'data.mini={MINI}',
        f'training.cuda={I_GPU}',
        f'training.batch_size={BATCH_SIZE}',
        f'training.epochs={EPOCHS}',
        f'training.num_workers={WORKERS}',
        f'training.optim.base_lr={BASE_LR}',
        f'training.wandb.log=False',
        f'training.wandb.name={EXP_NAME}',
        f'tracker_options.make_submission={SUBMISSION}',
        f'training.checkpoint_dir={CHECKPOINT_DIR}',
    ]


    cfg = hydra_read(overrides)
    main(cfg)
