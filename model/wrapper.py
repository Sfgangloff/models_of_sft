import pytorch_lightning as pl
import torch
import numpy as np
from .bipartite_model import GNN_SAT
from .GCN import GNNModel
import torch.nn as nn

class LitModel(pl.LightningModule):
   def __init__(self, cfg, lr, weight_decay, supervision_mode='sat'):
       super().__init__()
       self.save_hyperparameters(ignore=['model'])

       self.model = GNNModel(size_alphabet=len(cfg.data.alphabet),
                        embed_dim=cfg.model.embedding_dimensions,
                        num_hidden=cfg.model.hidden_dimensions)
       self.lr = cfg.train.lr
       self.weight_decay = weight_decay
       self.num_iters = cfg.model.number_iterations
       self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)       
        
   def training_step(self, batch, batch_idx):
       outputs = self.model(batch, self.num_iters)
       loss = self.loss_fn(outputs, batch.y.squeeze(1).long())
       self.log('train_loss', loss, prog_bar=True)
       return loss

   def validation_step(self, batch, batch_idx):
       outputs = self.model(batch, self.num_iters)
       loss = self.loss_fn(outputs, batch.y.squeeze(1).long())
       self.log('val_loss', loss, prog_bar=True)      
       return loss
   
   def configure_optimizers(self):
       return torch.optim.Adam(
           self.parameters(),
           lr=self.lr,
           weight_decay=self.weight_decay
       )
   