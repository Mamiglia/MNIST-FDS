import hydra
from omegaconf import DictConfig
import lightning as lit
from base import Net
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import wandb
from pytorch_lightning.loggers import WandbLogger
from dataset.loader import load_dataset

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Initialize Wandb logger
    wandb_logger = WandbLogger(project=cfg.wandb.project, name=cfg.wandb.name)

    train_loader, val_loader, test_loader = load_dataset(**cfg.dataset)

    model = Net(**cfg.net)

    trainer = lit.Trainer(logger=wandb_logger, **cfg.trainer)

    # Assuming you have a DataLoader `train_loader` and `val_loader`
    trainer.fit(model, train_loader, val_loader)
    
    # Assuming you have a DataLoader `test_loader`
    trainer.test(model, test_loader)
    


if __name__ == "__main__":
    main()