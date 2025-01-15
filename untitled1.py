#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:16:42 2025

@author: gaussian
"""

import torch
import lightning as L



ckpt_path = ""
checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
print(checkpoint["hyper_parameters"])
