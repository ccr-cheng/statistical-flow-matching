import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from utils import get_optimizer, get_scheduler
from . import get_flow_model
from .ema import ExponentialMovingAverage


class Text8Module(pl.LightningModule):
    def __init__(self, **config):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)
        self.model = get_flow_model(self.hparams.model, self.hparams.encoder)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.hparams.train.ema_decay)
        self.TEXT8_CHARS = list("_abcdefghijklmnopqrstuvwxyz")

    def forward(self, x):
        return self.model(x)

    def char_ids_to_str(self, char_ids) -> str:
        """Decode a 1D sequence of character IDs to a string."""
        return ''.join([self.TEXT8_CHARS[i] for i in char_ids])

    def batch_to_str(self, text_batch) -> list[str]:
        """Decode a batch of character IDs to a list of strings."""
        return [self.char_ids_to_str(row_char_ids) for row_char_ids in text_batch]

    def training_step(self, batch, batch_idx):
        """Perform a single training step on a batch of data from the training set."""
        loss = self.model.get_loss(*batch)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], logger=True)
        return loss

    def on_train_epoch_start(self) -> None:
        """Called at the beginning of the training epoch."""
        self.model.train()
        self.ema.to(self.device)

    def on_train_epoch_end(self):
        """Called at the end of the training epoch with the data from all training steps."""
        self.log('train_loss_epoch', self.trainer.callback_metrics['train_loss'], sync_dist=True)

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step on a batch of data from the validation set."""
        loss = self.model.get_loss(*batch)
        self.log('val_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    @rank_zero_only
    @torch.no_grad()
    def generate_text(self):
        traj = self.model.sample('euler', self.hparams.sample.n_sample, self.hparams.sample.n_step, self.device)
        txt = self.batch_to_str(traj.argmax(-1).tolist())
        for i, t in enumerate(txt):
            self.logger.experiment.add_text(f'sample{i}', t, self.trainer.global_step)

    def on_validation_epoch_start(self) -> None:
        """Called at the beginning of the validation epoch."""
        self.model.eval()
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())

    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch with the data from all validation steps."""
        self.log('val_loss_epoch', self.trainer.callback_metrics['val_loss'], logger=True, sync_dist=True)
        self.generate_text()
        self.ema.restore(self.model.parameters())
        self.model.train()

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs['optimizer'] if 'optimizer' in kwargs else args[2]
        if self.trainer.global_step < self.hparams.train.lr_warmup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / float(self.hparams.train.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.train.optimizer.lr
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.model.parameters())

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = get_optimizer(self.hparams.train.optimizer, self.model)
        scheduler = get_scheduler(self.hparams.train.scheduler, optimizer)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': 'val_loss',
                        'interval': 'step',
                        'frequency': self.hparams.train.val_check_interval,
                    },
                }
            else:
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',
                        'frequency': 1,
                    },
                }
        return {'optimizer': optimizer}
