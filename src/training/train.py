"""Loop de entrenamiento del U-Net de doble encoder (RGB + CHM)."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import TreeCrownTileDataset
from src.data.transforms import RandomFlipRotate
from src.losses.losses import CrownLoss
from src.models.unet_dual import build_model
from src.training.metrics import binary_iou

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    train_ds = TreeCrownTileDataset(
        data_cfg["train_dir"],
        rgb_scale=data_cfg.get("rgb_scale", 255.0),
        chm_max=data_cfg.get("chm_max", 40.0),
        transform=RandomFlipRotate(),
    )
    val_ds = TreeCrownTileDataset(
        data_cfg["val_dir"],
        rgb_scale=data_cfg.get("rgb_scale", 255.0),
        chm_max=data_cfg.get("chm_max", 40.0),
        transform=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"].get("batch_size", 8),
        shuffle=True,
        num_workers=config["training"].get("num_workers", 4),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"].get("batch_size", 8),
        shuffle=False,
        num_workers=config["training"].get("num_workers", 4),
    )
    return train_loader, val_loader


def train(config_path: str) -> None:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Usando device=%s", device)

    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    criterion = CrownLoss(**config.get("loss", {}))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"].get("lr", 1e-3))

    epochs = config["training"].get("epochs", 50)
    ckpt_dir = Path(config["training"].get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_iou = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            rgb = batch["rgb"].to(device)
            chm = batch["chm"].to(device)
            targets = {"mask": batch["mask"].to(device), "dist": batch["dist"].to(device)}

            optimizer.zero_grad()
            outputs = model(rgb, chm)
            losses = criterion(outputs, targets)
            losses["loss"].backward()
            optimizer.step()
            train_loss += losses["loss"].item()

        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(device)
                chm = batch["chm"].to(device)
                mask_gt = batch["mask"].to(device)
                outputs = model(rgb, chm)
                val_iou += binary_iou(outputs["mask_logits"], mask_gt)
        val_iou /= max(len(val_loader), 1)

        logger.info("Epoch %d/%d - train_loss=%.4f - val_iou=%.4f", epoch, epochs, train_loss, val_iou)

        torch.save(model.state_dict(), ckpt_dir / "last.pth")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            logger.info("Nuevo mejor modelo (val_iou=%.4f) guardado en %s", best_iou, ckpt_dir / "best.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
