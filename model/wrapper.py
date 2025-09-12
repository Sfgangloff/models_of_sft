# model/wrapper.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from hydra.utils import instantiate


class LitModel(pl.LightningModule):
    """
    Unified Lightning wrapper for:
      • Grid GCN models (tensor logits) with CE loss on batch.y (ignore_index=-100).
      • CNF bipartite models (GNN_SAT) returning dict with 'final_votes';
        uses BCE-with-logits on per-literal targets batch.y_l (masked by batch.y_l_mask).

    Detection:
      - If model(batch) is a dict containing 'final_votes' AND batch has y_l → CNF path.
      - Else → Grid path.

    Notes:
      - For CNF path, your DataLoader must use follow_batch=['x_l','x_c'].
      - For Grid path, y should be shape (num_nodes,1) with labels in {0..A} and -100 ignored.
    """

    def __init__(self, cfg, weight_decay: float):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        # Allow ${len:...} in Hydra configs
        OmegaConf.register_new_resolver("len", lambda x: len(x), replace=True)

        # Instantiate the chosen model from the config
        selected = cfg.model.parameters[cfg.model.name]
        self.model = instantiate(selected)

        # Optimizer hyperparams
        self.lr = cfg.train.lr
        self.weight_decay = weight_decay

        # Grid path CE loss (ignore masked targets)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    # ------------------------- helpers: CNF path ------------------------- #

    @staticmethod
    def _literal_bce_loss(
        logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        BCE-with-logits over literals, optionally masked.
        logits: (L,) float; y: (L,) in {0.,1.}; mask: (L,) bool.
        """
        if mask is not None:
            mask = mask.bool()
            if mask.sum() == 0:
                return logits.new_tensor(0.0)
            return F.binary_cross_entropy_with_logits(logits[mask], y[mask])
        return F.binary_cross_entropy_with_logits(logits, y)

    @staticmethod
    def _literal_accuracy(
        logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Literal-level accuracy (threshold 0.5).
        """
        with torch.no_grad():
            pred = (torch.sigmoid(logits) >= 0.5).float()
            if mask is not None:
                mask = mask.bool()
                if mask.sum() == 0:
                    return logits.new_tensor(float('nan'))
                return (pred[mask] == y[mask]).float().mean()
            return (pred == y).float().mean()

    # ------------------------- helpers: Grid path ------------------------ #

    def _grid_loss_and_acc(self, logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-entropy on grid nodes with ignore_index=-100; accuracy excludes ignored.
        logits: (N, C), y: (N,1) long with -100 masked.
        """
        y_long = y.squeeze(1).long()
        loss = self.ce(logits, y_long)

        with torch.no_grad():
            mask = y_long != -100
            if mask.any():
                pred = logits.argmax(dim=-1)
                acc = (pred[mask] == y_long[mask]).float().mean()
            else:
                acc = logits.new_tensor(float('nan'))
        return loss, acc

    # ---------------------- unified forward + loss ----------------------- #

    def _forward_and_metrics(self, batch):
        out = self.model(batch)

        # CNF bipartite path
        if isinstance(out, dict) and 'final_votes' in out and hasattr(batch, 'y_l'):
            logits = out['final_votes'].squeeze(-1)      # (L,)
            y = batch.y_l.float()
            mask = getattr(batch, 'y_l_mask', None)

            loss = self._literal_bce_loss(logits, y, mask)
            acc = self._literal_accuracy(logits, y, mask)
            path = 'cnf'
            return loss, acc, path

        # GCN path
        if hasattr(batch, 'y') and torch.is_tensor(out):
            loss, acc = self._grid_loss_and_acc(out, batch.y)
            path = 'grid'
            return loss, acc, path

        # If we get here, schema mismatch
        raise RuntimeError(
            "Could not determine training path. "
            "For CNF, model must return dict with 'final_votes' and batch must have y_l. "
            "For Grid, model must return tensor logits and batch must have y."
        )

    # ------------------------- Lightning hooks -------------------------- #

    def training_step(self, batch):
        loss, acc, path = self._forward_and_metrics(batch)
        self.log(f'{path}_train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{path}_train_acc',  acc,  prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        loss, acc, path = self._forward_and_metrics(batch)
        self.log(f'{path}_val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{path}_val_acc',  acc,  prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
