import torch
from data.sft_data import SFTDataModule, get_data
from model.wrapper import LitModel
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from torch_geometric.loader import DataLoader as GeoDataLoader
import logging
import numpy as np
import random


logging.basicConfig(level=logging.INFO)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_signature = f"GNN_SFT"
    logging.info(f"Model signature: {model_signature}")

    wrapped_model = LitModel(
        cfg=cfg,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    logger = WandbLogger(project="SFT", name=f"{model_signature}")

    data = get_data(cfg)

    datamodule = SFTDataModule(
        data=data,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.num_epochs,
        logger=logger,
        accelerator='cpu',
        devices=1,
        gradient_clip_val=cfg.train.gradient_clip_val,
    )

    trainer.fit(wrapped_model, datamodule)

    res = trainer.validate(wrapped_model, datamodule)
    #logger.log_metrics({'val_accuracy': res[0]['val_accuracy'],'param_range': data['param_range']})
    logging.info(f"Validation results: {res}")
    trainer.save_checkpoint(f"{model_signature}.ckpt")

if __name__ == '__main__':
    main()
     