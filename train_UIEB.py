import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyiqa
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from torchvision.models import vgg16, VGG16_Weights

from model.UWCNN import UWCNN

try:
    from model.UIEC2Net import UIEC2Net
except ImportError:
    UIEC2Net = None

try:
    from model.NU2Net import NU2Net
except ImportError:
    NU2Net = None

try:
    from model.FIVE_APLUS import FIVE_APLUSNet
except ImportError:
    FIVE_APLUSNet = None

try:
    from model.UTrans import UTrans
except ImportError:
    UTrans = None

from dataset.UIEB import UIEBDataset

try:
    from dataset.LSUI import LSUIDataset
except ImportError:
    LSUIDataset = None

from myutils.losses import *
from myutils.quality_refer import calc_psnr, calc_mse, calc_ssim, normalize_img


# =============================================================================
#  MATPLOTLIB TRAINING CURVE CALLBACK
# =============================================================================

class MatplotlibCurveCallback(Callback):
    def __init__(self, save_path="training_curves.png"):
        super().__init__()
        self.save_path  = save_path
        self.train_loss = []
        self.val_psnr   = []
        self.val_ssim   = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.train_loss.append(metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "psnr" in metrics:
            self.val_psnr.append(metrics["psnr"].item())
        if "ssim" in metrics:
            self.val_ssim.append(metrics["ssim"].item())
        if len(self.train_loss) > 0:
            self._save_plot()

    def _save_plot(self):
        epochs     = range(1, len(self.train_loss) + 1)
        val_epochs = range(1, len(self.val_psnr) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Training Curves — UWCNN on UIEB", fontsize=13, fontweight="bold")

        axes[0].plot(epochs, self.train_loss, "b-o", label="Train Loss",
                     markersize=4, linewidth=1.5)
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if self.val_psnr:
            axes[1].plot(val_epochs, self.val_psnr, "g-o", label="Val PSNR",
                         markersize=4, linewidth=1.5)
        axes[1].set_title("PSNR (dB)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("PSNR")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        if self.val_ssim:
            axes[2].plot(val_epochs, self.val_ssim, "m-o", label="Val SSIM",
                         markersize=4, linewidth=1.5)
        axes[2].set_title("SSIM")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("SSIM")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Curve] Saved -> {self.save_path}")


# =============================================================================
#  LIGHTNING MODULE
# =============================================================================

class TrainUIEModel(pl.LightningModule):
    def __init__(self, hparams):
        super(TrainUIEModel, self).__init__()

        model_zoos = {"UWCNN": UWCNN}
        if UIEC2Net:      model_zoos["UIEC2Net"]   = UIEC2Net
        if NU2Net:        model_zoos["NU2Net"]      = NU2Net
        if FIVE_APLUSNet: model_zoos["FIVE_APLUS"] = FIVE_APLUSNet
        if UTrans:        model_zoos["UTrans"]      = UTrans

        self.params       = hparams
        self.initlr       = self.params.initlr
        self.weight_decay = self.params.weight_decay
        self.lr_config    = self.params.lr_config

        self.ssim_loss = pyiqa.create_metric("ssim", as_loss=True)
        self.l1_loss   = MyLoss()
        self.char_loss = CharLoss()

        vgg_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval()
        vgg_model = vgg_model.features[:16]
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.per_loss = PerpetualLoss(vgg_model=vgg_model)

        self.val_ssim = pyiqa.create_metric("ssim")
        self.val_psnr = pyiqa.create_metric("psnr")

        if hparams.model_name not in model_zoos:
            raise ValueError(
                f"Model '{hparams.model_name}' not found. "
                f"Available: {list(model_zoos.keys())}"
            )
        self.model = model_zoos[hparams.model_name]()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.initlr,
            betas=[0.9, 0.999],
            weight_decay=self.weight_decay
        )
        if self.lr_config == "CyclicLR":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.initlr,
                max_lr=1.2 * self.initlr,
                cycle_momentum=False
            )
        elif self.lr_config == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.8
            )
        else:
            raise ValueError(f"Unknown lr_config: {self.lr_config}")
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat   = self.forward(x)
        loss    = self.l1_loss(y_hat, y) + 0.2 * self.per_loss(y_hat, y)
        self.log("train_loss", loss, sync_dist=True,
                 batch_size=x.shape[0], prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        assert x.shape[0] == 1

        y_hat      = self.forward(x)
        _, _, h, w = y.shape
        gt_img     = y[0].permute(1, 2, 0).detach().cpu().numpy()

        upsample = nn.UpsamplingBilinear2d((h, w))
        pred_img = upsample(normalize_img(y_hat))
        pred_img = pred_img[0].permute(1, 2, 0).detach().cpu().numpy()

        psnr = calc_psnr(pred_img, gt_img, is_for_torch=False)
        ssim = calc_ssim(pred_img, gt_img, is_for_torch=False)
        mse  = calc_mse(pred_img,  gt_img, is_for_torch=False)

        self.log("psnr", psnr, sync_dist=True, batch_size=1, prog_bar=True)
        self.log("ssim", ssim, sync_dist=True, batch_size=1, prog_bar=True)
        self.log("mse",  mse,  sync_dist=True, batch_size=1)

        return {"psnr": psnr, "ssim": ssim, "mse": mse}


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Training UIEB dataset")
    parser.add_argument("--model_name",   type=str,   default="UWCNN")
    parser.add_argument("--crop_size",    type=int,   default=256)
    parser.add_argument("--input_norm",   action="store_true")
    parser.add_argument("--epochs",       type=int,   default=110)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--initlr",       type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.000)
    parser.add_argument("--lr_config",    type=str,   default="CyclicLR")
    hparams = parser.parse_args()

    seed_everything(42)

    csv_logger = CSVLogger("logs/", name=hparams.model_name)

    RESUME          = True
    resume_ckpt     = "./checkpoints/UIEB/UWCNN.ckpt"
    checkpoint_path = "./checkpoints/UIEB/"
    os.makedirs(checkpoint_path, exist_ok=True)

    train_set = UIEBDataset(
        "./data/", train_flag=True, pred_flag=False,
        train_size=hparams.crop_size, input_norm=hparams.input_norm
    )
    test_set = UIEBDataset(
        "./data/", train_flag=False, pred_flag=False,
        train_size=hparams.crop_size, input_norm=hparams.input_norm
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=hparams.num_workers,
        pin_memory=True,
        persistent_workers=hparams.num_workers > 0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=hparams.num_workers,
        pin_memory=True,
        persistent_workers=hparams.num_workers > 0,
    )

    model = TrainUIEModel(hparams)
    if RESUME:
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"  Weights loaded from {resume_ckpt}")

    checkpoint_callback = ModelCheckpoint(
        monitor="psnr",
        dirpath=checkpoint_path,
        filename="epoch{epoch:02d}-psnr{psnr:.3f}-ssim{ssim:.3f}",
        auto_insert_metric_name=False,
        every_n_epochs=1,
        save_top_k=3,
        mode="max",
        save_last=True,
        save_weights_only=True
    )

    curve_callback = MatplotlibCurveCallback(save_path="training_curves.png")

    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        devices=[0],
        logger=csv_logger,
        accelerator="cuda",
        precision=16,
        callbacks=[checkpoint_callback, curve_callback],
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, test_loader,
    ckpt_path="./checkpoints/UIEB/last.ckpt")

if __name__ == "__main__":
    main()
