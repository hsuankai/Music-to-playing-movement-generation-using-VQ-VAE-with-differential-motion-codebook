import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vqvae.vqvae_1d.diffsound.encdec_yang import Sampler
from ..vqvae.vqvae_1d.encodec.modules.seanet import SEANetEncoder

'''  
'head': 0,
'neck': 1,
'torso': 2,
'r_hip': 3,
'l_hip': 4,
'r_shoulder': 5,
'r_elbow': 6,
'r_wrist': 7,
'r_finger': 8,
'l_shoulder': 9,
'l_elbow': 10,
'l_wrist': 11,
'l_finger': 12,
'r_knee': 13,
'r_ankle': 14,
'r_toe': 15,
'l_knee': 16,
'l_ankle': 17,
'l_toe': 18,
'VTOP': 19,
'VBOM': 20,
'BTOP': 21,
'BBOM': 22
'''

def get_normal(*shape, std=0.01):
    w = torch.empty(shape)
    nn.init.normal_(w, std=std)
    return w

def roll(x, n):
    return torch.cat((x[:, -n:], x[:, :-n]), dim=1)

class Transpose(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.transpose(1, 2)  

class MusicLanguagemodel(nn.Module):
    def __init__(self, input_dim, emb_dim, motion_codebook_size, max_len, **kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.motion_codebook_size = motion_codebook_size
        self.aggregate = kwargs['aggregate']
        self.vae = kwargs['vae']
        levels = kwargs['levels']
        
        # self.spf = kwargs['spf']
        # self.beat = kwargs['beat']
        # self.in_emb = nn.Sequential(
        #                             nn.Conv1d(hp.mel_features, hp.d_model, 3, 1, 1),
        #                             nn.GroupNorm(4, hp.d_model),
        #                             nn.ReLU(),
        #                             nn.Conv1d(hp.d_model, hp.d_model, 3, 1, 1),
        #                             nn.GroupNorm(4, hp.d_model),
        #                             nn.ReLU(),
        #                             )
        if self.aggregate:
            # original
            if 'VQ' in self.vae and 'R' not in self.vae:
                self.in_emb = nn.Sequential(
                                            nn.Conv2d(1, emb_dim, kernel_size=(3, 3)),
                                            nn.ELU(),
                                            nn.MaxPool2d(kernel_size=(1, 3)),
                                            nn.Dropout(0.15),
                                            nn.Conv2d(emb_dim, emb_dim, kernel_size=(1, 10)),
                                            nn.ELU(),
                                            nn.MaxPool2d((1, 3)),
                                            nn.Dropout(0.15),
                                            nn.Conv2d(emb_dim, emb_dim, kernel_size=(3, 3)),
                                            nn.ELU(),
                                            nn.MaxPool2d((1, 3)),
                                            nn.Dropout(0.15),
                                            nn.Conv2d(emb_dim, emb_dim, kernel_size=(3, 3), padding=(1, 1)),
                                            nn.ELU(),
                                            nn.MaxPool2d((1, 3)),
                                            nn.Dropout(0.15),
                                            )
            else:
                self.in_emb = nn.Sequential(
                                            nn.Conv2d(1, emb_dim, kernel_size=(3, 3)),
                                            nn.ELU(),
                                            nn.MaxPool2d(kernel_size=(1, 3)),
                                            nn.Dropout(0.15),
                                            nn.Conv2d(emb_dim, emb_dim, kernel_size=(1, 7)),
                                            nn.ELU(),
                                            nn.MaxPool2d((1, 3)),
                                            nn.Dropout(0.15),
                                            nn.Conv2d(emb_dim, emb_dim, kernel_size=(3, 3), padding=(0, 1)),
                                            nn.ELU(),
                                            nn.MaxPool2d((1, 3)),
                                            nn.Dropout(0.15),
                                            nn.Conv2d(emb_dim, emb_dim, kernel_size=(3, 3), padding=(1, 1)),
                                            nn.ELU(),
                                            nn.Dropout(0.15),
                                            )
            self.encoder = Sampler(emb_dim, emb_dim, kwargs['downsample_ratios'])
        else:
            # self.encoder = Sampler(input_dim, emb_dim, kwargs['downsample_ratios'])
            self.encoder = SEANetEncoder(channels = 128, dimension = 512, n_filters = 128, n_residual_layers = 1,
                                    ratios = [2, 2], activation = 'ELU', activation_params = {'alpha': 1.0},
                                    norm = 'weight_norm', norm_params = {}, kernel_size = 3,
                                    last_kernel_size = 3, residual_kernel_size = 3, dilation_base = 2, causal = False,
                                    pad_mode = 'constant', true_skip = True, compress = 2, lstm = 2)
            
       
        
        if 'VQ' in self.vae:
            self.proj_e = nn.Conv1d(emb_dim, motion_codebook_size, 1, 1, 0) #
        elif 'FSQ' in self.vae:
            motion_codebook_size = np.prod(levels)
            self.motion_codebook_size = motion_codebook_size
            self.proj_e = nn.Conv1d(emb_dim, motion_codebook_size, 3, 1, 1) #
            
        self.proj = nn.Sequential(nn.Conv1d(emb_dim, emb_dim, 3, 1, 1),
                                  nn.ReLU(),
                                  nn.Conv1d(emb_dim, 1, 1, 1, 0)
                                  )
    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        # assert len(x.shape) == 3
        if self.aggregate:
            N, T, S, C = x.shape
            x = x.reshape(-1, S, C)
            x = x.unsqueeze(1)
        else:
            x = x.permute(0,2,1).float()
        return x    
    
    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0,2,1)
        return x
    
    def forward(self, x_in, x_l, metrics):
        x = x_in['audio']
        # onset = x_in['onset']
        if self.aggregate:
            N, T, S, C = x.shape
        else:
            N, T, C = x.shape
        
        # Encode/Decode
        x = self.preprocess(x)
        # [NT x 64 x 5 x 128]
        if self.aggregate:
            x = self.in_emb(x) 
            NT, c, s, f = x.shape
            x = x.reshape(N, T, c, f)
            x = x.permute(0, 3, 2, 1)
            x = x.reshape((-1, c, T))
            x_l = x_l.transpose(1, 2) # (N, Q, T)
        x = self.encoder(x)
        # onset_pred = self.proj(x)
        x = self.proj_e(x)
        x = self.postprocess(x)
        motion_probability = F.softmax(x, dim=-1)
        x_l_pred = motion_probability.argmax(dim=-1)
        if 'FSQ' in self.vae:
            x_l = x_l.type(torch.LongTensor).to(x_l_pred.device)
        
        # Loss
        loss = F.cross_entropy(x.reshape(-1, self.motion_codebook_size), x_l.reshape(-1))   
        _, _, C = motion_probability.shape
        motion_probability = motion_probability.reshape((-1, C))
        x_l = x_l.reshape((-1))
        correct_probility = torch.stack([motion_probability[i, idx] for i, idx in enumerate(x_l)])
        # onset_loss = F.binary_cross_entropy_with_logits(onset_pred.transpose(1, 2), onset)
        # if 'VQ' in self.vae and 'R' not in self.vae:
        #     correct_probility = torch.stack([motion_probability[0, t, idx] for t, idx in enumerate(x_l[0])])
        # else:
        #     NQ, T, C = motion_probability.shape
        #     motion_probability = motion_probability.reshape((NQ*T, C))
        #     N, Q, T = x_l.shape # (2, 4, 37)
        #     x_l = x_l.reshape((-1))
        #     correct_probility = torch.stack([motion_probability[i, idx] for i, idx in enumerate(x_l)])
        
        # loss += onset_loss
        try:
            # metrics['onset_loss'].append(onset_loss)
            metrics['correct_probability'].append(correct_probility)
        except:
            # metrics['onset_loss'] = []
            metrics['correct_probability'] = []
            # metrics['onset_loss'].append(onset_loss)
            metrics['correct_probability'].append(correct_probility)
        
        if self.training:
            return loss, metrics, 
        else:
            return loss, metrics, x_l_pred
