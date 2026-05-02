import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .dataset import PairedUIEBDataset
from .metrics import psnr
from .model import build_nafnet


def main():
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(config["device"])

    dataset = PairedUIEBDataset(
        root=config["dataset"]["root"],
        split=args.split,
        image_size=config["dataset"]["image_size"],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = build_nafnet(config["model"]["nafnet_code_dir"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate"):
            underwater = batch["underwater"].to(device)
            target = batch["target"].to(device)
            name = batch["name"][0]

            prediction = model(underwater).clamp(0, 1)
            scores.append(psnr(prediction, target))

            save_image(prediction.cpu(), output_dir / name)

    average_psnr = sum(scores) / len(scores)
    print(f"Average PSNR on {args.split}: {average_psnr:.2f} dB")


def choose_device(requested_device):
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(requested_device)


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default="outputs/enhanced")
    return parser.parse_args()


if __name__ == "__main__":
    main()
