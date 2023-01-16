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
    
    DIR = '/home/rozenberszki/altay/DeepViewAgg_playground'
    ROOT = os.path.join(DIR, "..")
    sys.path.insert(0, ROOT)
    sys.path.insert(0, DIR)

    train = True



    DATA_ROOT="/mnt/hdd/datasets"                                       # set your dataset root directory, where the data was/will be downloaded
    EXP_NAME="dev-debug"                              # whatever suits your needs
    TASK="segmentation"
    MODELS_CONFIG=f"{TASK}/multimodal/adl4cv-scannet"                         # family of multimodal models using the sparseconv3d backbone
    MODEL_NAME="Res16UNet34-L3-intermediate-ade20k-interpolate-local-fusion"      # specific model name
    # MODEL_NAME = "base-local-fusion"
    DATASET_CONFIG=f"{TASK}/multimodal/scannet-sparse"
    #TRAINING="scannet_benchmark/minkowski-pretrained-pyramid-0"     # training configuration for discriminative learning rate on the model for pyramid ones
    TRAINING = "scannet_benchmark/minkowski-pretrained-5"            # training configuration for discriminative learning rate on the model for a network with a single 2d layer   
    EPOCHS=2
    TRAINVAL=False                                                          # True to train on Train+Val (eg before submission)
    BATCH_SIZE=2                                                    
    WORKERS=8                                                               # adapt to your machine
    BASE_LR=0.1                                                             # initial learning rate
    LR_SCHEDULER='constant'                                                 # learning rate scheduler for 60 epochs
    EVAL_FREQUENCY=1                                                        # frequency at which metrics will be computed on Val. The less the faster the training but the less points on your validation curves
    SUBMISSION=False                                                        # True if you want to generate files for a submission to the KITTI-360 3D semantic segmentation benchmark
    CHECKPOINT_DIR=""                                                       # optional path to an already-existing checkpoint. If provided, the training will resume where it was left
    TENSORBOARD = False

    overrides = [
        'task=segmentation',
        f'data={DATASET_CONFIG}',
        f'models={MODELS_CONFIG}',
        f'model_name={MODEL_NAME}',
        f'task={TASK}',
        f'training={TRAINING}',
        f'lr_scheduler={LR_SCHEDULER}',
        f'eval_frequency={EVAL_FREQUENCY}',
        f'data.dataroot={DATA_ROOT}',
        #f'data.train_is_trainval={TRAINVAL}',
        f'training.cuda={I_GPU}',
        f'training.batch_size={BATCH_SIZE}',
        f'training.epochs={EPOCHS}',
        f'training.num_workers={WORKERS}',
        f'training.optim.base_lr={BASE_LR}',
        f'training.wandb.log=False',
        f'training.wandb.name={EXP_NAME}',
        f'tracker_options.make_submission={SUBMISSION}',
        f'training.checkpoint_dir={CHECKPOINT_DIR}',
        f'training.tensorboard.log={TENSORBOARD}',
    ]


    cfg = hydra_read(overrides)
    main(cfg)
