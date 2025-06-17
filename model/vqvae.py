import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from .bottleneck import  BottleneckBlock
from .encdec import Encoder1d, Decoder1d


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
class MotionVQVAE(L.LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        fps = cfg.fps
        model_params = cfg.motionvqvae.model
        loss_weight = cfg.motionvqvae.loss_weight
        emb_dim = model_params.ch * model_params[f"fps{fps}"].ch_mult[-1]
        self.rec, self.vel, self.acc, self.commit, self.commit1, self.commit2, self.commit3 = loss_weight.values()
        self.codebook_size = model_params.codebook_size
        self.encoder = Encoder1d(ch_mult=model_params[f"fps{fps}"].ch_mult, **model_params)
        self.decoder = Decoder1d(ch_mult=model_params[f"fps{fps}"].ch_mult, **model_params)
        self.bottleneck = nn.ModuleList([BottleneckBlock(self.codebook_size[i], emb_dim, 0.99) for i in range(len(self.codebook_size))])
        self.refine = nn.Conv1d(emb_dim*len(self.codebook_size), emb_dim, kernel_size=1, padding=0)
        self.example_input_array = torch.zeros(1, int(fps)*20, 69)  # optional
        
    def quantization(self, x):
        N, C, T_ = x.shape
        commit_losses = 0
        x_ls = []
        x_ds = []
        for i, bottleneck in enumerate(self.bottleneck):
            # vanilla quantization
            if i == 0:
                x_l, x_d, commit_loss, metrics = bottleneck(x, update_k=self.training)
                x_l = x_l.view(N, T_)
                x_d = x_d.view(N, T_, -1)
                x_ls.append(x_l)
                x_ds.append(x_d)
                commit_loss = commit_loss * self.commit1
                commit_losses += commit_loss
                metrics["commit_loss"] = commit_loss
                for k, v in metrics.items():
                    metrics[k] = v.float()
            # 1st order quantization X2:XT_ refinement
            elif i == 1:
                x_diff = x[..., 1:] - x[..., :-1]
                x_diff_l, x_diff_d, commit_diff_loss, metrics_diff = bottleneck(x_diff, update_k=self.training)
                x_diff_l = x_diff_l.view(N, T_-1)
                x_diff_d = x_diff_d.view(N, T_-1, -1)
                x_ls.append(x_diff_l)
                x_ds.append(x_diff_d)
                commit_diff_loss = commit_diff_loss * self.commit2
                commit_losses += commit_diff_loss
                metrics["commit_diff_loss"] = commit_diff_loss
                for k, v in metrics_diff.items():
                    metrics[k + "_diff"] = v.float()
            # 2nd order quantization X3:XT_ refinement
            elif i == 2:
                x_diff2 = x_diff[..., 1:] - x_diff[..., :-1]
                x_diff2_l, x_diff2_d, commit_diff2_loss, metrics_diff2 = bottleneck(x_diff2, update_k=self.training)
                x_diff2_l = x_diff2_l.view(N, T_-2)
                x_diff2_d = x_diff2_d.view(N, T_-2, -1)
                x_ls.append(x_diff2_l)
                x_ds.append(x_diff2_d)
                commit_diff2_loss = commit_diff2_loss * self.commit3
                commit_losses += commit_diff2_loss
                metrics["commit_diff2_loss"] = commit_diff2_loss
                for k, v in metrics_diff2.items():
                    metrics[k + "_diff2"] = v.float()
        commit_losses /= len(self.codebook_size)
        return x_ds, x_ls, commit_losses, metrics
    
    def refine_quantize(self, x_ds):
        for i in range(len(self.bottleneck)):
            if i == 0:
                x_d = x_ds[i]
                
            elif i == 1:
                x_diff_d = x_ds[i]
                # cumulative sum 1st order differential quantized representation
                x_d_refine1 = torch.cat((x_d[:, 0:1], x_diff_d), dim=1) # (x1, x2-x1, x3-x2, ..., xt_-xt_-1)
                x_d_refine1 = torch.cumsum(x_d_refine1, dim=1)[:, 1:] # (x2, ..., xt_)
                # stack vanilla quantized representation and 1st order differential quantized representation    
                x_d_refine1 = torch.cat((x_d[:, 1:], x_d_refine1), dim=-1)
                x_d_refine1 = self.refine(x_d_refine1.transpose(1, 2)).transpose(1, 2)
                x_d = torch.cat((x_d[:, 0:1], x_d_refine1), dim=1)
                
            elif i == 2:
                x_diff2_d = x_ds[i]
                # cumulative sum 2st order differential quantized representation
                x_d_refine2 = torch.cat((x_diff_d[:, 0:1], x_diff2_d), dim=1) # ((x2-x1), (x3-x2)-(x2-x1), ..., (xt_-xt_-1)-(xt_-1-xt_-2))
                x_d_refine2 = torch.cumsum(x_d_refine2, dim=1) # (x2-x1, x3-x2, ..., xt_-xt_-1)
                x_d_refine2 = torch.cat((x_d[:, 0:1], x_d_refine2), dim=1) # (x1, x2-x1, x3-x2, ..., xt_-xt_-1)
                x_d_refine2 = torch.cumsum(x_d_refine2, dim=1)[:, 2:] # (x3, ..., xt_)
                # stack vanilla quantized representation, 1st order differential quantized representation, and 2nd order differential quantized representation            
                pad1 = torch.zeros_like(x_d_refine1)[:, 0:1]
                pad2 = torch.zeros_like(x_d_refine2)[:, 0:2]
                x_d_refine1 = torch.cat((pad1, x_d_refine1), dim=1)
                x_d_refine2 = torch.cat((pad2, x_d_refine2), dim=1)
                x_d_refine = torch.cat((x_d, x_d_refine1, x_d_refine2), dim=-1)
                x_d = self.refine(x_d_refine)
        x_d = x_d.transpose(1, 2)
        return x_d
    
    def encode(self, x):
        N, T, C = x.shape
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x_ds, x_ls, commit_losses, metrics = self.quantization(x)
        x_d = self.refine_quantize(x_ds)
        # Passthrough
        x_d = x + (x_d - x).detach()
        return x_d, x_ls, commit_losses, metrics

    def decode(self, x_d, T):
        x_out = self.decoder(x_d, T)
        x_out = x_out.permute(0, 2, 1)
        return x_out
    
    def decode_code(self, x_ls, T):
        x_ds = []
        for x_l, bottleneck in zip(x_ls, self.bottleneck):
            x_d = bottleneck.decode(x_l).transpose(1, 2)
            x_ds.append(x_d)
        x_d = self.refine_quantize(x_ds)        
        # Decode
        x_out = self.decode(x_d, T)
        return x_out
    
    def forward(self, x):
        N, T, C = x.shape
        # Encode and quantization
        x_d, x_ls, commit_losses, metrics = self.encode(x)
        # Decode
        x_out = self.decode(x_d, T)
        return x_out, x_ls, commit_losses, metrics
    
    def loss(self, commit_losses, x_out, x_target):
        recons_loss = F.l1_loss(x_out, x_target)
        velocity_loss = F.l1_loss(x_out[:, 1:] - x_out[:, :-1], x_target[:, 1:] - x_target[:, :-1])
        acceleration_loss =  F.l1_loss(x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1], x_target[:, 2:] + x_target[:, :-2] - 2 * x_target[:, 1:-1])
        monitor_loss = recons_loss + velocity_loss + commit_losses
        loss = self.rec * recons_loss + self.vel * velocity_loss + self.acc * acceleration_loss +  self.commit * commit_losses
        loss_metrics = dict(
            monitor_loss=monitor_loss,
            loss=loss,
            recons_loss=recons_loss,
            velocity_loss=velocity_loss,
            acceleration_loss=acceleration_loss)
        return loss, loss_metrics
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch["keypoints"]
        x_target = batch["keypoints"]
        x_out, x_ls, commit_losses, metrics = self(x)
       
        # Loss
        loss, loss_metrics = self.loss(commit_losses, x_out, x_target)
        metrics.update(loss_metrics)
        log_metrics = {}
        for k, v in metrics.items():
            log_metrics[f"train_{k}"] = v
        self.log_dict(log_metrics, prog_bar=True, on_step=False, on_epoch=True,  sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch["keypoints"]
        x_target = batch["keypoints"]
        x_out, x_ls, commit_losses, metrics = self(x)
       
        # Loss
        loss, loss_metrics = self.loss(commit_losses, x_out, x_target)
        metrics.update(loss_metrics)
        log_metrics = {}
        for k, v in metrics.items():
            log_metrics[f"val_{k}"] = v
        self.log_dict(log_metrics, prog_bar=True, on_step=False, on_epoch=True,  sync_dist=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.cfg.optim)
        return optimizer