import hydra
from omegaconf import DictConfig
import lightning as lit
from net.base import Net
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import wandb
from lightning.pytorch.loggers import WandbLogger
from dataset.cifar import *

import lightning.pytorch.callbacks as cb

from omegaconf import OmegaConf

import os

def load_data(num_workers, batch_size, val_ratio, seed):
    train, val, test = load_cifar10_data(val_ratio, seed)
    train_loader = to_dataloader(train, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    val_loader = to_dataloader(val, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    test_loader = to_dataloader(test, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

@hydra.main(config_path="../cfg", config_name="base")
def main(cfg: DictConfig):
    lit.seed_everything(cfg.seed)
    wandb_logger = WandbLogger(**cfg.wandb)

    model = Net(cfg.net)
    
    train, val, test = load_data(**cfg.dataset)

    trainer = lit.Trainer(logger=wandb_logger, callbacks=[
        cb.EarlyStopping(
            monitor="val_acc",
            patience=3,
            verbose=True,
            mode="max", min_delta=1e-2
        )], **cfg.trainer)

    print('Training...')
    trainer.fit(model, train, val)
    
    print('Testing...')
    trainer.test(model, test)
    
    hyperparams_dict = OmegaConf.to_container(cfg, resolve=True)
    hyperparams_dict["info"] = {  # type: ignore
        "num_params": get_num_params(model),
    }
    wandb_logger.log_hyperparams(hyperparams_dict)  # type: ignore

def get_num_params(module):
    """
    Returns the number of parameters in a Lightning module.
    
    Args:
        module (lightning.pytorch.LightningModule): The Lightning module to get the number of parameters for.
    
    Returns:
        int: The number of parameters in the module.
    """
    total_params = sum(p.numel() for p in module.parameters() )
    return total_params



if __name__ == "__main__":
    os.environ['HYDRA_FULL_ERROR'] = "1"
    main()