# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import time
import typing as tp

import math
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from audiocraft import models
from audiocraft import optim

from ..vqvae.vqvae_1d.diffsound.encdec_yang import Sampler
from audiocraft.solvers.builders import get_ema

from ..vqvae.vqvae_1d.encodec.modules.seanet import SEANetEncoder

class MusicSolver(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()
        
        self.cfg = cfg
        # raw, trained, pre_trained
        self.music_vqvae: tp.Optional[nn.Module] = None
        
   def forward(self, batch):
       if 