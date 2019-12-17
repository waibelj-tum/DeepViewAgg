import torch
from torch import nn
import torch.nn.functional as F
from typing import Any

from models.unet_base import UnetBasedModel


class SegmentationModel(UnetBasedModel):
    def __init__(self, option, model_name, num_classes, modules):
        """Initialize this model class.
        Parameters:
            opt -- training/test options
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        UnetBasedModel.__init__(self, option, model_name, num_classes,
                                modules)  # call the initialization method of UnetBasedModel

        nn = option.mlp_cls.nn
        self.dropout = option.mlp_cls.get('dropout')
        self.lin1 = torch.nn.Linear(nn[0], nn[1])
        self.lin2 = torch.nn.Linear(nn[2], nn[3])
        self.lin3 = torch.nn.Linear(nn[4], num_classes)

        self.loss_names = ['loss_seg']

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input = (data.x, data.pos, data.batch)
        self.labels = data.y

    def forward(self) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        x, _, _ = self.model(self.input)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        self.output = F.log_softmax(x, dim=-1)
        return self.output

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results

        self.loss_seg = F.nll_loss(self.output, self.labels) + self.get_internal_losses()
        self.loss_seg.backward()       # calculate gradients of network G w.r.t. loss_G
