import argparse
import pickle
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

import yaml
import numpy as np
import torch

from metrics import (
    compute_maje,
    compute_mad,
    compute_hellinger_distance,
    compute_apd,
    compute_attack_f1,
)
from model.vqvae import MotionVQVAE
from model.audio2motion import Audio2Motion
from utils import AttrDict, audio_aggregate

# -------------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Test audio2motion model")
parser.add_argument(
    "--fps",
    type=int,
    default=120,
    help="Frame rate for loading test data and checkpoints (e.g., 30, 60, 120)",
)
parser.add_argument(
    "--model",
    choices=("motionvqvae", "audio2motion"),
    default="audio2motion",
    help="Which model to test",
)
parser.add_argument(
    "--device",
    default="cuda",
    help="Device to run on (e.g., 'cpu' or 'cuda')",
)
args = parser.parse_args()

FPS = args.fps
DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------
# Dataset Settings 
# -------------------------------------------------------------------------
DATASET_DICT = {
    "yv": {
        "vid": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
           "pieces_vio": [
                "ba1", "ba3", "ba4", "be4", "be5", "be6", "be7", "be8",
                "el1", "de1", "me4", "mo4", "mo5", "de2",
        ],
    },
    "ev": {
        "vid": ["01", "02", "03", "04", "05"],
           "pieces_vio": [
                "ba1", "ba3", "ba4", "be4", "be8",
                "el1", "de1", "me4", "mo4", "mo5", "de2",
        ],
    },
}
TEST_PIECES = ["be4", "be5", "be6", "be7"]

# -------------------------------------------------------------------------
# Load test data and config
# -------------------------------------------------------------------------
with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
cfg = AttrDict.from_nested_dicts(cfg)
cfg.fps = f"{args.fps}"

data_dir = Path("data")
data_file = f"test_fps{FPS}.pkl"
data_path = data_dir / data_file
if not data_path.exists():
    data_dir.mkdir(parents=True, exist_ok=True)
    # repo_id="hkkao/mosa_motion", path_in_repo="audio2motion/30.ckpt"
    downloaded = hf_hub_download(
        repo_id="hkkao/mosa_motion",
        filename=f"fps{cfg.fps}.zip",
        local_dir=str(data_dir),
        repo_type="dataset"
    )
    # hf_hub_download returns the full path to the downloaded file:
    zip_path = Path(downloaded)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    # Clean up temporary file
    zip_path.unlink(missing_ok=True)
with open(data_path, "rb") as f:
    data = pickle.load(f)
    
# -------------------------------------------------------------------------
# Initialize model
# -------------------------------------------------------------------------
test_model = args.model
ckpt_dir = Path(cfg[test_model]["ckpt_path"])
ckpt_file = f"fps{cfg.fps}.ckpt"                     # e.g. "fps30.ckpt"
ckpt_path = ckpt_dir / ckpt_file

if not ckpt_path.exists():
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # repo_id="hkkao/mosa_motion", path_in_repo="audio2motion/30.ckpt"
    downloaded = hf_hub_download(
        repo_id="hkkao/mosa_motion",
        filename=f"{test_model}/{ckpt_file}",
        local_dir="checkpoint",
        repo_type="model"
        )

if test_model == "motionvqvae":
    model = MotionVQVAE.load_from_checkpoint(str(ckpt_path), cfg=cfg).to(DEVICE)
elif test_model == "audio2motion":
    vqvae = MotionVQVAE(cfg)
    model = Audio2Motion.load_from_checkpoint(str(ckpt_path), cfg=cfg, vqvae=vqvae).to(DEVICE)
else:
    raise ValueError(f"Unsupported model: {test_model}")
    
model.eval()

# -------------------------------------------------------------------------
# Prepare metric accumulators
# -------------------------------------------------------------------------
full_maje_vals = []
rh_maje_vals = []
rh_mad_vals = []
hellinger_vals = []
pred_diversity_vals = []
targ_diversity_vals = []
attack_f1_vals = []

results: dict = {}

# -------------------------------------------------------------------------
# Main evaluation loop
# -------------------------------------------------------------------------
for dataset in ["yv", "ev"]:
    results[dataset] = {}
    for piece in DATASET_DICT[dataset]["pieces_vio"]:
        results[dataset][piece] = {}
        for vid in DATASET_DICT[dataset]["vid"]:
            results[dataset][piece][vid] = {}

            if piece not in TEST_PIECES:
                continue

            print(f"Evaluating Dataset: {dataset}, Piece: {piece}, Vid: {vid}")
            
            aud = data[dataset][piece][vid]["aud"][0]
            targ_kp = data[dataset][piece][vid]["keypoints"][0]
            kp_mean = data[dataset][piece][vid]["keypoints_mean"][0]
            kp_std = data[dataset][piece][vid]["keypoints_std"][0]
            
            if test_model=="audio2motion" and cfg.data.audio.aggr_len is not None:
                aud = audio_aggregate(aud, cfg.data.audio.aggr_len)
            aud = torch.from_numpy(aud[np.newaxis]).float().to(DEVICE)
            targ_kp = torch.from_numpy(targ_kp[np.newaxis]).float().to(DEVICE)
            
            with torch.no_grad():
                if test_model == "motionvqvae":
                    pred_kp = model.forward(targ_kp)
                elif test_model == "audio2motion":
                    pred_kp = model.generate(aud)
                
            pred_kp = pred_kp[0].cpu().numpy()
            targ_kp = targ_kp[0].cpu().numpy()
            pred_kp = pred_kp * (kp_std + 1e-8) + kp_mean
            targ_kp = targ_kp * (kp_std + 1e-8) + kp_mean
            pred_kp = pred_kp.reshape(-1, pred_kp.shape[-1] // 3, 3)
            targ_kp = targ_kp.reshape(-1, targ_kp.shape[-1] // 3, 3)
            
            full_maje, rh_maje = compute_maje(pred_kp, targ_kp)
            rh_mad = compute_mad(pred_kp, targ_kp)
            hellinger_dist = compute_hellinger_distance(
                pred_kp, targ_kp, visualize=False, fps=FPS
            )
            pred_div, targ_div = compute_apd(pred_kp, targ_kp, fps=FPS)
            attack_f1_score = compute_attack_f1(pred_kp, targ_kp, fps=FPS)

            full_maje_vals.append(full_maje)
            rh_maje_vals.append(rh_maje)
            rh_mad_vals.append(rh_mad)
            hellinger_vals.append(hellinger_dist)
            pred_diversity_vals.append(pred_div)
            targ_diversity_vals.append(targ_div)
            attack_f1_vals.append(attack_f1_score)

            results[dataset][piece][vid]["pred_keypoints"] = pred_kp
            results[dataset][piece][vid]["targ_keypoints"] = targ_kp

# -------------------------------------------------------------------------
# Compute overall averages
# -------------------------------------------------------------------------
avg_full_maje = np.mean(full_maje_vals)
avg_rh_maje = np.mean(rh_maje_vals)
avg_rh_mad = np.mean(rh_mad_vals)
avg_hellinger = np.mean(hellinger_vals)
avg_pred_div = np.mean(pred_diversity_vals)
avg_targ_div = np.mean(targ_diversity_vals)
avg_attack_f1 = np.mean(attack_f1_vals)

# -------------------------------------------------------------------------
# Print summary
# -------------------------------------------------------------------------
print(f"\n=== Evaluation Summary (FPS={FPS}) ===")
print(f"Full MAJE:           {avg_full_maje:.3e}")
print(f"Right-Hand MAJE:     {avg_rh_maje:.3e}")
print(f"Right-Hand MAD:      {avg_rh_mad:.3e}")
print(f"Hellinger Distance:  {avg_hellinger:.3e}")
print(f"Pred Diversity (APD):{avg_pred_div:.3e}")
print(f"Targ Diversity (APD):{avg_targ_div:.3e}")
print(f"Attack F1 Score:     {avg_attack_f1:.3f}")

# -------------------------------------------------------------------------
# Save results dictionary
# -------------------------------------------------------------------------
save_dir = Path(f"results/fps{FPS}/keypoints/{test_model}")
save_dir.mkdir(parents=True, exist_ok=True)
output_path =  save_dir / f"fps{FPS}.pkl"
output_path.parent.mkdir(exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump(results, f)