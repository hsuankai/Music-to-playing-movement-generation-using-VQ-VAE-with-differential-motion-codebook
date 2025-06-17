import argparse
import os
import pickle
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download

from model.vqvae import MotionVQVAE
from model.audio2motion import Audio2Motion
from utils import AttrDict
from visualize.animation import plot_animation


# -------------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Inference your own audio")
parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="Frame rate for loading test data and checkpoints (e.g., 30, 60, 120)",
)
parser.add_argument(
    "--audio",
    type=str,
    help="Specify audio path",
)
parser.add_argument(
    "--device",
    default="cuda",
    help="Device to run on (e.g., 'cpu' or 'cuda')",
)
parser.add_argument(
    "--save_path",
    default="results/keypoints",
    help="Save path of predicted points",
)
parser.add_argument(
    "--plot_path",
    default="results/animation",
    help="Plot path of 3D animation",
)
args = parser.parse_args()

FPS = args.fps
DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------
# Load test data and config
# -------------------------------------------------------------------------
with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
cfg = AttrDict.from_nested_dicts(cfg)

# Override cfg.fps to match the integer FPS argument
cfg.fps = f"{FPS}"

# -------------------------------------------------------------------------
# Initialize model
# -------------------------------------------------------------------------
vqvae = MotionVQVAE(cfg)
ckpt_dir = Path(cfg.audio2motion.ckpt_path)       
ckpt_file = f"fps{FPS}.ckpt"                     # e.g. "fps30.ckpt"
ckpt_path = ckpt_dir / ckpt_file
if not ckpt_path.exists():
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # repo_id="hkkao/mosa_motion", path_in_repo="audio2motion/30.ckpt"
    downloaded = hf_hub_download(
        repo_id="hkkao/mosa_motion",
        filename=f"audio2motion/{ckpt_file}",
        local_dir="checkpoint",
        repo_type="model"
    )
    # hf_hub_download returns the full path to the downloaded file:
model = Audio2Motion.load_from_checkpoint(
    str(ckpt_path), cfg=cfg, vqvae=vqvae
).to(DEVICE)

model.eval()
with torch.no_grad():
    pred = model.inference(args.audio, DEVICE, FPS)

save_dir = Path(args.save_path)
save_dir.mkdir(parents=True, exist_ok=True)
audio_name = args.audio.split("/")[-1]
audio_name, ext = os.path.splitext(audio_name)
save_path = args.save_path + "/" + f"{audio_name}.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(pred, f)
plot_animation(args.audio, args.plot_path, pred)