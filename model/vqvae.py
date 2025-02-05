import torch
import torch.nn as nn
import torch.nn.functional as F
from .bottleneck import  BottleneckBlock
import lightning as L
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
    def __init__(self, cfg, fps, loss_weight, in_channels=128, codebook_num=2, codebook_size=[512, 512], emb_dim=512, gn=32, act="swish",
                 ch=128, num_res_blocks=2, dropout=0.1, resamp_with_conv=True, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        model_params = cfg.motionvqvae["model"]
        loss_weight = cfg.motionvqvae["loss_weight"]
        self.commit, self.vel, self.acc = loss_weight.values()
        self.codebook_num = codebook_num
        self.encoder = Encoder1d(in_channels, emb_dim, ch)
        self.decoder = Decoder1d(in_channels, emb_dim, ch)
        self.bottleneck = nn.ParameterList([BottleneckBlock(args.codebook_size[i], emb_dim, 0.99) for i in range(args.codebook_num)])
        self.refine = nn.Linear(args.emb_dim*args.codebook_num, args.emb_dim)
    
    def quantization(self, x):
        N, T_, C = x.shape
        commit_losses = []
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
                commit_losses.append(commit_loss)
                metrics["commit_loss"] = commit_loss
            # 1st order quantization X2:XT_ refinement
            elif i == 1:
                x_diff = x[..., 1:] - x[..., :-1]
                x_diff_l, x_diff_d, commit_diff_loss, metrics_diff = bottleneck(x_diff, update_k=self.training)
                x_diff_l = x_diff_l.view(N, T_-1)
                x_diff_d = x_diff_d.view(N, T_-1, -1)
                x_ls.append(x_diff_l)
                x_ds.append(x_diff_d)
                commit_losses.append(commit_diff_loss)
                metrics["commit_diff_loss"] = commit_diff_loss
                for k, v in metrics_diff.items():
                    metrics[k + "_diff"] = v
            # 2nd order quantization X3:XT_ refinement
            elif i == 2:
                x_diff2 = x_diff[..., 1:] - x_diff[..., :-1]
                x_diff2_l, x_diff2_d, commit_diff2_loss, metrics_diff2 = bottleneck(x_diff2, update_k=self.training)
                x_diff2_l = x_diff2_l.view(N, T_-2)
                x_diff2_d = x_diff2_d.view(N, T_-2, -1)
                x_ls.append(x_diff2_l)
                x_ds.append(x_diff2_d)
                commit_losses.append(commit_diff2_loss)
                metrics["commit_diff2_loss"] = commit_diff2_loss
                for k, v in metrics_diff2.items():
                    metrics[k + "_diff2"] = v
        return x_ds, x_ls, commit_losses, metrics
    
    def refine_quantize(self, x_ds):
        for i in range(self.codebook_num):
            if i == 0:
                x_d = x_ds[i]
            elif i == 1:
                x_diff_d = x_ds[i]
                # cumulative sum 1st order differential quantized representation
                x_d_refine1 = torch.cat((x_d[:, 0:1], x_diff_d), dim=1) # (x1, x2-x1, x3-x2, ..., xt_-xt_-1)
                x_d_refine1 = torch.cumsum(x_d_refine1, dim=1)[:, 1:] # (x2, ..., xt_)
                # stack vanilla quantized representation and 1st order differential quantized representation    
                x_d_refine1 = torch.cat((x_d[:, 1:], x_d_refine1), dim=-1)
                x_d_refine1 = self.refine(x_d_refine1)
                x_d = torch.cat((x_d[:, 0:1], x_d_refine1), dim=1)
                x_d = x_d.transpose(1, 2)
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
    
    def forward(self, x, x_target):
        N, T, C = x.shape
        # Encode and quantization
        x_d, x_ls, commit_losses, metrics = self.encode(x)
        # Decode
        x_out = self.decode(x_d, T)
        return x_out, x_ls, commit_losses, metrics
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch["keypoints"]
        x_target = batch["keypoints"]
        x_out, x_ls, commit_losses, metrics = self(x, x_target)
       
        # Loss
        commit_losses = torch.mean(commit_losses)
        recons_loss = F.l1_loss(x_out, x_target)
        velocity_loss = F.l1_loss(x_out[:, 1:] - x_out[:, :-1], x_target[:, 1:] - x_target[:, :-1])
        acceleration_loss =  F.l1_loss(x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1], x_target[:, 2:] + x_target[:, :-2] - 2 * x_target[:, 1:-1])
        loss = (1-self.commit) * (recons_loss + self.vel * velocity_loss + self.acc * acceleration_loss) +  self.commit * commit_losses
        metrics.update(dict(
            loss=loss,
            recons_loss=recons_loss,
            velocity_loss=velocity_loss,
            acceleration_loss=acceleration_loss))
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch["keypoints"]
        x_target = batch["keypoints"]
        x_out, x_ls, commit_losses, metrics = self(x, x_target)
       
        # Loss
        commit_losses = torch.mean(commit_losses)
        recons_loss = F.l1_loss(x_out, x_target)
        velocity_loss = F.l1_loss(x_out[:, 1:] - x_out[:, :-1], x_target[:, 1:] - x_target[:, :-1])
        acceleration_loss =  F.l1_loss(x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1], x_target[:, 2:] + x_target[:, :-2] - 2 * x_target[:, 1:-1])
        loss = (1-self.commit) * (recons_loss + self.vel * velocity_loss + self.acc * acceleration_loss) +  self.commit * commit_losses
        metrics.update(dict(
            loss=loss,
            recons_loss=recons_loss,
            velocity_loss=velocity_loss,
            acceleration_loss=acceleration_loss))
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters, lr=1e-4, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.1)
        return optimizer
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        # Implement your own custom logic to clip gradients
        # You can call `self.clip_gradients` with your settings:
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
        )
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)
    
    
    
    # def on_train_epoch_end(self):
    #     # do something with all training_step outputs, for example:
    #     epoch_mean = torch.stack(self.training_step_outputs).mean()
    #     self.log("training_epoch_mean", epoch_mean)
    #     # free up the memory
    #     self.training_step_outputs.clear()
