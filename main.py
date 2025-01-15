#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:04:11 2025

@author: gaussian
"""

from types import SimpleNamespace
from torch.utils.data import DataLoader
from dataset import audio_skeleton_dataset
import lightning as L
from model.vqvae import MotionVQVAE
import yaml
L.pytorch.seed_everything(42, workers=True)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


# from argparse import ArgumentParser


# def main(hparams):
#     model = LightningModule()
#     trainer = Trainer(accelerator=hparams.accelerator, devices=hparams.devices)
#     trainer.fit(model)


# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--accelerator", default=None)
#     parser.add_argument("--devices", default=None)
#     args = parser.parse_args()

#     main(args)

with open("config/config.yaml", 'r') as f:
    cfg = yaml.full_load(f)
cfg = SimpleNamespace(**cfg)


train_model = "MotionVQVAE"
train_loader = DataLoader(audio_skeleton_dataset(cfg.data, split="train", train_model=train_model))
valid_loader = DataLoader(audio_skeleton_dataset(cfg.data, split="valid", train_model=train_model))


# # train with both splits
if train_model == "MotionVQVAE":
    model = MotionVQVAE(cfg)
elif train_model == "MotionVQVAE":
    model = MotionVQVAE(cfg.audioenc)
trainer = L.Trainer(**cfg.Trainer, callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
trainer.fit(model, train_loader, valid_loader)