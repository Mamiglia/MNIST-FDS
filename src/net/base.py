import torch
import lightning as lit
from torchmetrics import Accuracy

import torch.nn as nn
import torch.nn.functional as F

from cnn import ConvBlock, FeedForwardBlock

class Net(lit.LightningModule):
    def __init__(self, 
                 input_channels=1, 
                 num_classes=10, 
                 depth=1,
                 conv_params: dict = {},
                 optimizer_params: dict = {}):
                 
        super(Net, self).__init__()
        self.save_hyperparameters()
        self.optimizer_params = optimizer_params


        self.conv = [ConvBlock(input_channels, **conv_params)]
        self.conv += [ConvBlock(**conv_params) for _ in range(depth-1)]
        self.conv = nn.Sequential(*self.conv)
        
        self.classifier = nn.LazyLinear(num_classes)        


    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optimizer_params)
        return optimizer