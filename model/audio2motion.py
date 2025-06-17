import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from .bottleneck import BottleneckBlock
import lightning as L
from .encdec import AudEncoder
from pathlib import Path
from typing import Tuple, Optional, Union
from utils import audio_aggregate

"""  
'head': 0, 
'neck': 1, 'torso': 2,
'r_hip': 3, 'l_hip': 4,
'r_shoulder': 5, 'r_elbow': 6, 'r_wrist': 7, 'r_finger': 8,
'l_shoulder': 9, 'l_elbow': 10, 'l_wrist': 11, 'l_finger': 12,
'r_knee': 13, 'r_ankle': 14, 'r_toe': 15,
'l_knee': 16, 'l_ankle': 17, 'l_toe': 18,
'VTOP': 19, 'VBOM': 20,
'BTOP': 21, 'BBOM': 22
"""

# define the LightningModule
class Audio2Motion(L.LightningModule):
    def __init__(self, cfg, vqvae, **kwargs):
        super().__init__()
        fps = cfg.fps
        model_params = cfg.audio2motion.model
        in_channels = model_params.in_channels
        dp = model_params['dropout']
        self.cfg = cfg
        self.register_buffer("keypoints_mean", torch.zeros(69))
        self.register_buffer("keypoints_std", torch.zeros(69))
        
        if 'keypoints_mean' in cfg.keys():
            keypoints_mean = torch.from_numpy(cfg.keypoints_mean).float()
            keypoints_std = torch.from_numpy(cfg.keypoints_std).float()
            self.register_buffer("keypoints_mean", keypoints_mean)
            self.register_buffer("keypoints_std", keypoints_std)
            
        self.codebook_size = model_params.codebook_size
        self.aggr_len = cfg.data.audio.aggr_len
        if self.aggr_len != None:
            window_kernel = {1: [1, 1], 3: [3, 1], 5: [3, 3], 7: [3, 5], 9: [3, 9]}[self.aggr_len]
            self.in_emb = nn.Sequential(
                                        nn.Conv2d(1, in_channels, kernel_size=(window_kernel[0], 3)),
                                        nn.ELU(),
                                        nn.MaxPool2d(kernel_size=(1, 3)),
                                        nn.Dropout(dp),
                                        nn.Conv2d(in_channels, in_channels, kernel_size=(1, 10)),
                                        nn.ELU(),
                                        nn.MaxPool2d((1, 3)),
                                        nn.Dropout(dp),
                                        nn.Conv2d(in_channels, in_channels, kernel_size=(window_kernel[1], 3)),
                                        nn.ELU(),
                                        nn.MaxPool2d((1, 3)),
                                        nn.Dropout(dp),
                                        nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=(1, 1)),
                                        nn.ELU(),
                                        nn.MaxPool2d((1, 3)),
                                        nn.Dropout(dp),
                                        )
            self.example_input_array = torch.zeros(1, int(fps)*20, self.aggr_len, 128)  # optional
        else:
            self.example_input_array = torch.zeros(1, int(fps)*20, 128)  # optional
        
        self.vqvae = vqvae
        self.encoder = AudEncoder(ch_mult=model_params[f"fps{fps}"].ch_mult, **model_params)
        self.vqvae.eval()
    
    def load_and_normalize_audio(self,
        audio_path: str,
        sr: Optional[int] = 44100,
        target_db: Optional[int] = -20,
    ) -> Tuple[np.ndarray, int]:
        """
        Load an audio file, resample if needed, and normalize its loudness to target dBFS.
    
        Args:
            audio_path:   Path to the audio file.
            sr:           Target sampling rate. If None, keep native.
            target_db:  Desired average loudness in dBFS (default: -20 dBFS).
    
        Returns:
            (y, sr):
                y:  1D float32 numpy array in range [-1.0, +1.0].
                sr: Sampling rate after loading.
        """
        # 1) Load (with librosa, returns float32 in [-1, +1])
        y, sr = librosa.load(audio_path, sr=sr)
            
        # 2) Compute current RMS and dBFS
        rms = np.sqrt(np.mean(y**2))
        scalar = 10 ** (target_db / 20) / (rms + 1e-9)
        y = y * scalar
        return y, sr
    
    def compute_mel_spectrogram(self,
        y: np.ndarray,
        sr: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
    ) -> torch.Tensor:
        """
        Convert a 1D numpy audio array into a Mel-spectrogram tensor.
    
        Args:
            y:   1D float32 numpy array in [-1.0, +1.0].
            sr:         Sampling rate (e.g., 44100).
            n_mels:     Number of Mel bands.
            n_fft:      FFT window size.
            hop_length: Hop length between frames.
    
        Returns:
            mel_spec_db:  numpy array of shape (1, n_mels, T), in dB scale.
        """
    
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            htk=True
        )
        # Convert to log scale (dB). Using max reference to scale.
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db.T
    
    def audio_preprocessing(self,
        audio_path: Union[str, Path],
        device: torch.device,
        fps: int,
        n_mels: int = 128,
    ) -> torch.tensor:
        """
        Full pipeline: Load audio, normalize to target dBFS, compute Mel-spectrogram
        with parameters chosen according to `fps`.
    
        Args:
            audio_path:   Path to the input audio file.
            device:       torch.device ("cuda" or "cpu") where model is loaded.
            fps:          Frame rate of the keypoint sequence (30, 60, or 120).
            target_dBFS:  Desired normalization loudness in dBFS.
            n_mels:       Number of Mel bands (default=128).
    
        Returns:
            mel_spec_db:  numpy array of shape (1, n_mels, T), in dB scale.
        """
        # 1) Determine n_fft, hop_length, and sample_rate based on fps
        sample_rate = 44100
        n_fft = 4096
        if fps == 30:
            hop_length = 1470
        elif fps == 60:
            hop_length = 735
        elif fps == 120:
            hop_length = 367
        else:
            raise ValueError(f"Unsupported fps: {fps}. Choose 30, 60, or 120.")
    
        # 2) Load & normalize audio at chosen sample_rate
        y, sr = self.load_and_normalize_audio(audio_path, sample_rate)
        assert sr == sample_rate, f"Expected sr={sample_rate}, got sr={sr}"
    
        # 3) Compute Mel-spectrogram with these parameters
        mel_spec_db = self.compute_mel_spectrogram(
            y=y,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )  # shape: (1, T, n_mels)
    
        # 4) Prepare tensor for model: add batch dimension → (1, 1, n_mels, T)
        mean = np.mean(mel_spec_db, axis=0)
        std = np.std(mel_spec_db, axis=0)
        mel_spec_db = (mel_spec_db - mean) / (std + 1E-8)
        return mel_spec_db
    
    def inference(self, src, device, fps):
        # Nomalize audio and extract mel spectrogram
        x_aud = self.audio_preprocessing(src, device=device, fps=fps)
        # Aggregate audio if aggregate length is set 
        if self.aggr_len is not None:
            x_aud = audio_aggregate(x_aud, aggr_len=self.aggr_len)
        # Convert to torch tensor and add channel dimension: (1, n_mels, T)
        x_aud =  torch.from_numpy(x_aud).unsqueeze(0).float().to(device)
        pred = self.generate(x_aud)
        pred = pred[0].cpu().numpy()
        keypoints_mean = self.keypoints_mean.cpu().numpy()
        keypoints_std = self.keypoints_std.cpu().numpy()
        pred = pred * (keypoints_std + 1e-8) + keypoints_mean
        pred = pred.reshape(-1, pred.shape[-1] // 3, 3)
        return pred
        
    def generate(self, x_aud):
        T = x_aud.shape[1]
        logits = self(x_aud)
        logits = [torch.argmax(logit, dim=-1) for logit in logits]
        keypoints = self.vqvae.decode_code(logits, T)
        return keypoints
        
    def preprocess(self, x):
        if self.aggr_len != None:
            N, T, S, F = x.shape
            x = x.reshape(-1, S, F)
            x = x.unsqueeze(1)
        else:
            x = x.permute(0, 2, 1)
        return x    
        
    def forward(self, x_aud):
        N, T, *_ = x_aud.shape
        x = self.preprocess(x_aud)
        if self.aggr_len != None:
            x = self.in_emb(x) 
            NT, C, S, F = x.shape
            x = x.reshape(N, T, C)
            x = x.transpose(1, 2)
        logits = self.encoder(x)
        logits = [logit.transpose(1, 2) for logit in logits]
        return logits
    
    def metrics(self, logits, x_targets):
        logits = [logit.reshape(-1, codebook_size) for logit, codebook_size in zip(logits, self.codebook_size)]
        x_targets = [x_target.reshape(-1) for x_target in x_targets]
        x_probs = [F.softmax(logit, dim=-1) for logit in logits]
        # x_tokens = [x_prob.argmax(dim=-1) for x_prob in x_probs]
        ce_losses = []
        for logit, x_target, in zip(logits, x_targets):
            ce_loss = F.cross_entropy(logit, x_target)
            ce_losses.append(ce_loss)
        loss = torch.mean(torch.stack(ce_losses))
        token_accs = []
        for x_prob, x_target in zip(x_probs, x_targets):
            tokens_acc = torch.stack([x_prob[i, index] for i, index in enumerate(x_target)])
            tokens_acc = torch.mean(tokens_acc)
            token_accs.append(tokens_acc)
        token_accs = torch.stack(token_accs)
        token_acc = torch.mean(token_accs)
        return loss, token_acc, ce_losses, token_accs
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x_aud = batch["aud"]
        x_keypoints = batch["keypoints"]
        # extract paired motion code
        _, x_targets, _, _ = self.vqvae.encode(x_keypoints)
        # predict motion code
        logits = self(x_aud)
        loss, token_acc, ce_losses, token_accs = self.metrics(logits, x_targets)
        log_metrics = {"train_loss": loss}
        for i, ce_loss in enumerate(ce_losses):
            log_metrics[f"train_ce_loss{i}"] = ce_loss
        for i, token_acc in enumerate(token_accs):
            log_metrics[f"train_token_acc{i}"] = token_acc
        self.log_dict(log_metrics, prog_bar=True, on_step=False, on_epoch=True,  sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x_aud = batch["aud"]
        x_keypoints = batch["keypoints"]
        # extract paired motion code
        _, x_targets, _, _ = self.vqvae.encode(x_keypoints)
        # predict motion code
        logits = self(x_aud)
        loss, token_acc, ce_losses, token_accs = self.metrics(logits, x_targets)
        log_metrics = {"val_loss": loss}
        for i, ce_loss in enumerate(ce_losses):
            log_metrics[f"val_ce_loss{i}"] = ce_loss
        for i, token_acc in enumerate(token_accs):
            log_metrics[f"val_token_acc{i}"] = token_acc
        self.log_dict(log_metrics, prog_bar=True, on_step=False, on_epoch=True,  sync_dist=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.1)
        return optimizer
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)