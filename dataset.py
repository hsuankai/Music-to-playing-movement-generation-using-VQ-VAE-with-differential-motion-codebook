import pickle
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional
from huggingface_hub import hf_hub_download

import torch
from torch.utils.data import Dataset
from utils import audio_aggregate


class AudioSkeletonDataset(Dataset):
    def __init__(
        self,
        cfg: Any,
        split: str = "train",
        model: str = "motionvqvae",
        fps: int = 30
    ) -> None:
        """
        Args:
            cfg: configuration namespace/dict, must have `train_path`, `audio.aggr_len`
            split: one of "train" or "val"
            train_model: "motionvqvae" or "audio2motion"
        """
        self.model = model
        self.aggr_len: Optional[int] = cfg.audio.aggr_len

        # build path and load pickle
        data_dir = Path("data")
        data_file = f"train_fps{fps}.pkl"
        data_path = data_dir / data_file
        if not data_path.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            # repo_id="hkkao/mosa_motion", path_in_repo="audio2motion/30.ckpt"
            downloaded = hf_hub_download(
                repo_id="hkkao/mosa_motion",
                filename=f"fps{fps}.zip",
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
            data = pickle.load(f)[split]

        # always present
        self.keypoints = data["keypoints"]
        self.len = len(data["aud"])
        self.keypoints_mean: float = data["keypoints_mean"]
        self.keypoints_std: float = data["keypoints_std"]

        # only for audio2motion
        if model == "audio2motion":
            self.audios = data["aud"]

        # clean up
        del data

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # always return keypoints tensor
        kp = torch.from_numpy(self.keypoints[idx]).float()
        item: Dict[str, torch.Tensor] = {"keypoints": kp}

        if self.model == "audio2motion":
            aud = self.audios[idx]
            if self.aggr_len is not None:
                aud = audio_aggregate(aud, aggr_len=self.aggr_len)
            item["aud"] = torch.from_numpy(aud).float()

        return item
