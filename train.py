import yaml
from argparse import ArgumentParser
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

from dataset import AudioSkeletonDataset
from model.vqvae import MotionVQVAE
from model.audio2motion import Audio2Motion
from utils import AttrDict, DelayedEarlyStopping

L.pytorch.seed_everything(42, workers=True)


def parse_args():
    parser = ArgumentParser(description="Train MotionVQVAE or Audio2Motion models")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate for loading test data and checkpoints (e.g., 30, 60, 120)",
    )
    parser.add_argument(
        "--train_model",
        choices=("motionvqvae", "audio2motion"),
        default="audio2motion",
        help="Which model to train",
    )
    parser.add_argument(
        "--accelerator",
        default="cuda",
        help="Training accelerator (e.g., 'cpu', 'cuda')",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use",
    )
    return parser.parse_args()


def load_config(path: Path):
    with path.open("r") as f:
        data = yaml.full_load(f)
    return AttrDict.from_nested_dicts(data)


def build_dataloaders(cfg, train_model: str):
    train_dataset = AudioSkeletonDataset(cfg.data, split="train", train_model=train_model)
    val_dataset = AudioSkeletonDataset(cfg.data, split="val", train_model=train_model)

    batch_size = cfg.data.batch_size
    num_workers = cfg.data.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Propagate dataset statistics into cfg
    cfg.keypoints_mean = train_dataset.keypoints_mean
    cfg.keypoints_std = train_dataset.keypoints_std

    return train_loader, val_loader


def select_model(cfg, train_model: str):
    if train_model == "motionvqvae":
        return MotionVQVAE(cfg)

    # audio2motion: load pretrained VQ-VAE
    ckpt = Path(cfg.motionvqvae.ckpt_path) / f"fps{cfg.fps}.ckpt"
    vqvae = MotionVQVAE.load_from_checkpoint(str(ckpt), cfg=cfg)
    return Audio2Motion(cfg, vqvae)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    FPS = args.fps

    # Override cfg.fps to match the integer FPS argument
    cfg.fps = f"{FPS}"

    
    train_loader, val_loader = build_dataloaders(cfg, args.train_model)
    model = select_model(cfg, args.train_model)
    
    # Callbacks
    early_stop = DelayedEarlyStopping(**cfg.earlystopping)
    checkpoint = ModelCheckpoint(
        filename=f"fps{cfg.fps}-{{epoch}}-{{val_loss:.2f}}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )
    summary = ModelSummary(max_depth=-1)
    callbacks = [early_stop, checkpoint, summary]

    trainer = L.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        accelerator=args.accelerator,
        devices=args.devices,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
