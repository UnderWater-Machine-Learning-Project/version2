import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PairedUIEBDataset
from .model import (
    build_nafnet,
    freeze_backbone_for_warmup,
    load_pretrained_weights,
    trainable_parameters,
    unfreeze_all,
)


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    device = choose_device(config["device"])
    train_loader = make_loader(config, split="train", shuffle=True)
    val_loader = make_loader(config, split="val", shuffle=False)

    model = build_nafnet(config["model"]["nafnet_code_dir"]).to(device)
    model = load_pretrained_weights(
        model,
        config["model"]["pretrained_weights"],
        device,
    )

    loss_fn = nn.L1Loss()
    save_dir = Path(config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Stage 1: train decoder/head while most layers are frozen")
    freeze_backbone_for_warmup(model)
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=config["training"]["head_lr"],
    )
    run_epochs(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        config["training"]["head_only_epochs"],
        save_dir / "stage1_head_only.pth",
    )

    print("Stage 2: unfreeze all layers and fine-tune with a lower learning rate")
    unfreeze_all(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["full_lr"],
    )
    run_epochs(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        config["training"]["full_finetune_epochs"],
        save_dir / "stage2_full_finetune.pth",
    )


def run_epochs(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    device,
    total_epochs,
    checkpoint_path,
):
    best_val_loss = float("inf")

    for epoch in range(1, total_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate(model, val_loader, loss_fn, device)

        print(
            f"epoch {epoch:03d}/{total_epochs:03d} "
            f"train_l1={train_loss:.5f} val_l1={val_loss:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"saved best checkpoint: {checkpoint_path}")


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        underwater = batch["underwater"].to(device)
        target = batch["target"].to(device)

        prediction = model(underwater)
        loss = loss_fn(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(loader, desc="val", leave=False):
        underwater = batch["underwater"].to(device)
        target = batch["target"].to(device)

        prediction = model(underwater)
        loss = loss_fn(prediction, target)

        total_loss += loss.item()

    return total_loss / len(loader)


def make_loader(config, split, shuffle):
    dataset = PairedUIEBDataset(
        root=config["dataset"]["root"],
        split=split,
        image_size=config["dataset"]["image_size"],
    )
    return DataLoader(
        dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=True,
    )


def choose_device(requested_device):
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(requested_device)


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
