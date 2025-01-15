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
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from audiocraft import models
from audiocraft import optim
from audiocraft.solvers.builders import get_ema

from torchmetrics import Accuracy

# from ..vqvae.vqvae_1d.encodec.modules.seanet import SEANetEncoder


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

class MotionGenSolver(nn.Module):
    """Solver for MusicGen training task.

    Used in: https://arxiv.org/abs/2306.05284
    """

    def __init__(self, data_args, cfg: omegaconf.DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_args = data_args
        self.device = cfg.device
        self.motion_vqvae: nn.Module
        self.music_vqvae: tp.Optional[nn.Module] = None
        self.model: tp.Optional[models.LMModel] = None
        self.music_aggregate = cfg.music.aggregate
        self.motion_aggregate = cfg.motion.aggregate
        
        # instantiate model & exponential moving average on the model
        self.build_model()
        if self.cfg.lm.music=='raw':
            self.buil_mel_transform()
        self.accuracy = Accuracy(task="multiclass", num_classes=self.cfg.motion[self.cfg.motion.arch].codebook_size).to(self.device)
        # self._ema_sources: nn.ModuleDict = nn.ModuleDict()
        # self.ema: tp.Optional[optim.ModuleDictEMA] = None
        # if not self.aggregate:
        #     self.register_ema('model')
        # else:
        #     self.register_ema('model', 'encoder', 'in_emb')
        # self.ema = get_ema(self._ema_sources, self.cfg.optim.ema)
        
        # easier access to sampling parameters
        self.generation_params = {
            'use_sampling': self.cfg.generate.lm.use_sampling, # True
            'temp': self.cfg.generate.lm.temp, # 1.0
            'top_k': self.cfg.generate.lm.top_k, # 0
            'top_p': self.cfg.generate.lm.top_p, # 0.0
        }
        self._best_metric_name: tp.Optional[str] = 'ce'
    
    def build_vqvae(self, cfg, modal_type, train=False, fine_tune=False):
        data_args = self.data_args
        dataset_name = self.data_args['dataset_name']
        modal_cfg = cfg[modal_type]
        data_format = modal_cfg.format # wav, mel, sequence, graph
        if data_format=='graph':
            from our.model.vqvae.graph.graph1.vqvae import VQVAEGraph
            # model = VQVAEGraph(3, hp.motion_codebook_size, hp.motion_emb_dim, **kwargs)
            # model.to(self.device)
            # models['vqvae'] = model
        else:
            try: 
                input_dim = cfg.data[data_format]['input_dim']
            except:
                if 'tnua' in dataset_name:
                    input_dim = cfg.data.motion['tnua']['input_dim']
                else:
                    input_dim = cfg.data.motion['mosa']['input_dim']
            from our.model.vqvae.sequence.vqvae import VQVAE
            if self.cfg.lm.music=='train' and modal_type=='music':
                if fine_tune:
                    model = VQVAE(modal_cfg, input_dim, modal_type)
                else:
                    model = VQVAE(modal_cfg, input_dim, modal_type)
                    if self.music_aggregate:
                        self.in_emb = model.in_emb
                    model = model.encoder
            else:
                model = VQVAE(modal_cfg, input_dim, modal_type)
            
            model.to(self.device)
        
        
        if not train or fine_tune:
            prefix = f'{data_args["dataset_name"]}/fps{data_args["fps"]}_{data_args["seq_len"]}'
            checkpoint_dir = f'our/checkpoint/{prefix}/vqvae/{modal_type}/{data_format}{cfg[modal_type].shape}/'
            if fine_tune:
                if 'ex' in f'{cfg[modal_type].name}':
                    if self.music_aggregate:
                        checkpoint_dir += f'{cfg[modal_type].name}'.rstrip('_ex').rstrip('_finetune').rstrip(('_aggregate'+ str(self.cfg.music.aggr_length))) + '_aggregate_ex.pth'
                    else:
                        checkpoint_dir += f'{cfg[modal_type].name}'.rstrip('_ex').rstrip('_finetune') + '_ex.pth'
                else:   
                    checkpoint_dir += f'{cfg[modal_type].name}'.rstrip('_finetune') + '.pth'
            else:
                checkpoint_dir += f'{cfg[modal_type].name}.pth'
            checkpoint = torch.load(checkpoint_dir)['model_state_dict']
            model.load_state_dict(checkpoint)
            print(f'load {modal_type} vqvae {cfg[modal_type].name} successfully')
            
            
            if not fine_tune:
                for param in model.parameters():
                    param.requires_grad = False
            else:
                if self.music_aggregate:
                    self.in_emb = model.in_emb
                if cfg.music.first_norm:
                    self.first_norm = model.first_norm
                
                model = model.encoder
                # print(model)
                # for i, (name, module) in enumerate(model.named_children()):
                #     if i<2:
                #         for param in module.parameters():
                #             param.requires_grad = False
                #         print(i, f"Freezing module: {name}")
            
                
        return model
        
    def build_model(self) -> None:
        """Instantiate models and optimizer."""
        # we can potentially not use all quantizers with which the EnCodec model was trained
        # (e.g. we trained the model with quantizers dropout)
        
        # build motion vqvae
        self.motion_vqvae = self.build_vqvae(self.cfg, 'motion')
        
        
        # build music vqvae
        from our.model.vqvae.sequence.vqvae import VQVAE
        from our.model.vqvae.sequence.diffsound.encdec import Encoder1diff
        from our.model.vqvae.sequence.yang.encdec import Sampler
        from omegaconf import OmegaConf,open_dict
        model_cfg = self.cfg.music['diffsound']
        with open_dict(model_cfg):
            model_cfg.act = 'swish'
        if self.cfg.lm.music!='raw':
            train = True if self.cfg.lm.music=='train' else False
            # self.music_vqvae = self.build_vqvae(self.cfg, 'music', train, fine_tune=self.cfg.lm.fine_tune)
            self.music_vqvae = Encoder1diff(in_channels=128, out_ch=model_cfg.codebook_size,
                                            **model_cfg)
            if self.music_aggregate:
                model = VQVAE(self.cfg.music, 128, 'music')
                self.in_emb = model.in_emb
            # self.music_vqvae = Sampler(128, model_cfg.emb_dim, model_cfg.downsample_ratios, quant_conv=model_cfg.quant_conv)
            music_dim = self.cfg.music[self.cfg.music.arch].emb_dim
            codebook_size = self.cfg.motion[self.cfg.motion.arch].codebook_size
            transformer_dim = self.cfg.transformer_lm.dim
            
            if self.cfg.lm.music=='pretrained':
                # if self.cfg.music[self.cfg.music.arch].vq_type=='VQ':
                #     self.embed = nn.Embedding(codebook_size, transformer_dim)
                #     self.proj = nn.Conv1d(transformer_dim, transformer_dim, 1)
                #     print('use token')
                # else:
                self.proj = nn.Conv1d(music_dim, transformer_dim, 1)
                    
            elif self.cfg.lm.music=='train':
                if self.cfg.lm.lm_use:
                    self.proj = nn.Conv1d(music_dim, transformer_dim, 1)
                # else:
                #     self.proj = nn.Conv1d(music_dim, codebook_size, 3, padding=1)
                #     self.proj_diff = nn.Conv1d(music_dim, codebook_size, 3, padding=1)
        else:
            self.input_norm = nn.LayerNorm(128)
            self.proj = nn.Conv1d(128, 512, 1)
            
        # instantiate LM model
        if self.cfg.lm.lm_use:
            self.model = models.builders.get_lm_model(self.cfg).to(self.device)
    
    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        # assert len(x.shape) == 3
        if self.music_aggregate:
            N, T, S, C = x.shape
            x = x.reshape(-1, S, C)
            x = x.unsqueeze(1)
        else:
            x = x.permute(0,2,1)
            if self.cfg.music.first_norm:
                x = self.first_norm(x)
        return x    
    
    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0,2,1)
        return x
    
    
    def _prepare_tokens_and_attributes(
        self, music: torch.Tensor, motion_tokens = None,
    ) -> tp.Tuple[dict, torch.Tensor, torch.Tensor]:
        """Prepare input batchs for language model training.

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]): Input batch with audio tensor of shape [B, C, T]
                and corresponding metadata as SegmentWithAttributes (with B items).
            check_synchronization_points (bool): Whether to check for synchronization points slowing down training.
        Returns:
            Condition tensors (dict[str, any]): Preprocessed condition attributes.
            Tokens (torch.Tensor): Audio tokens from compression model, of shape [B, K, T_s],
                with B the batch size, K the number of codebooks, T_s the token timesteps.
            Padding mask (torch.Tensor): Mask with valid positions in the tokens tensor, of shape [B, K, T_s].
        """
        
        """Compute pairs of `(embedding, mask)` using the configured conditioners and the tokenized representations.
        The output is for example:
        {
            "genre": (torch.Tensor([B, 1, D_genre]), torch.Tensor([B, 1])),
            "description": (torch.Tensor([B, T_desc, D_desc]), torch.Tensor([B, T_desc])),
            ...
        }
        """
        # Encode/Decode
        if self.cfg.lm.music!='raw':
            if self.cfg.lm.music=='train':
                N, T, *_ = music.shape
                x = self.preprocess(music)
                if self.music_aggregate:
                    x = self.in_emb(x)
                    x = x.reshape(N, T, 128).transpose(1, 2)
                embeds = self.music_vqvae(x)
                
            elif self.cfg.lm.music=='pretrained':
                if self.cfg.music[self.cfg.motion.arch].vq_type=='VQ':
                    x_l, _, = self.music_vqvae.encode(music)
                    embeds = self.embed(x_l).transpose(1, 2)
                else:
                    embeds = self.music_vqvae.encoder(music.transpose(1, 2))
                    embeds = self.proj(embeds)
        else:
            embeds = self.wav_to_mel(music)
        
        logits = []
        masks = []
        for e in embeds:
            mask = torch.ones_like(e[..., 0], dtype=torch.bool, device=e.device)
            e = (e * mask.unsqueeze(-1))
            logits.append(e.transpose(1, 2))
            masks.append(mask)
            condition_tensors = {'wav': (logits, masks)}
            
        return condition_tensors
    
    def forward(self, batch: dict,  metrics: dict) -> dict:
        """Perform one training or valid step on a given batch."""
        
        music = batch['audio'] # wav or mel  1. wav enc 2. mel enc 3. wav to mel
        motion = batch['keypoints']
        N, T, *_ = motion.shape
        keypoints_target = motion
        
        self.motion_vqvae.eval()
        if self.cfg.lm.music!='train':
            self.music_vqvae.eval()
        
        motion_tokens, _ = self.motion_vqvae.encode(motion)
        condition_tensors = self._prepare_tokens_and_attributes(music, motion_tokens)
        logits = condition_tensors['wav'][0]
        
        loss = []
        motion_tokens_preds = []
        correct_probabilities = []
        tokens_accs = []
        for logit, motion_token in zip(logits, motion_tokens):
            ce = F.cross_entropy(logit.reshape(-1, self.cfg.motion[self.cfg.motion.arch].codebook_size), motion_token.reshape(-1))
            loss.append(ce)
            motion_tokens_pred, correct_probability, tokens_acc = self.cal_acc(logit, motion_token)
            motion_tokens_preds.append(motion_tokens_pred)
            correct_probabilities.append(correct_probability)
            tokens_accs.append(tokens_acc)
        loss = torch.stack(loss).mean()
        
        # keypoint loss
        keypoints_pred = self.motion_vqvae.decode(motion_tokens_preds, T)
        keypoints_loss = F.l1_loss(keypoints_pred, keypoints_target)
        try:
            metrics['keypoints_loss'].append(keypoints_loss)
        except:
            metrics['keypoints_loss'] = []
            metrics['keypoints_loss'].append(keypoints_loss)
        
        i = 0
        for correct_probability, tokens_acc in zip(correct_probabilities, tokens_accs):
            if i==0:
                try:
                    metrics['correct_probability'].append(correct_probability)
                    metrics['tokens_acc'].append(tokens_acc)
                except:
                    metrics['correct_probability'] = []
                    metrics['correct_probability'].append(correct_probability)
                    metrics['tokens_acc'] = []
                    metrics['tokens_acc'].append(tokens_acc)
            elif i==1:
                try:
                    metrics['correct_probability_diff'].append(correct_probability)
                    metrics['tokens_diff_acc'].append(tokens_acc)
                except:
                    metrics['correct_probability_diff'] = []
                    metrics['correct_probability_diff'].append(correct_probability)
                    metrics['tokens_acc_diff'] = []
                    metrics['tokens_acc_diff'].append(tokens_acc)
            elif i==2:
                try:
                    metrics['correct_probability_diff2'].append(correct_probability)
                    metrics['tokens_acc_diff2'].append(tokens_acc)
                except:
                    metrics['correct_probability_diff2'] = []
                    metrics['correct_probability_diff2'].append(correct_probability)
                    metrics['tokens_acc_diff2'] = []
                    metrics['tokens_acc_diff2'].append(tokens_acc)
            i+=1
            
        metrics = dict(sorted(metrics.items()))
        if self.training:
            return loss, metrics, 
        else:
            return loss, metrics, motion_tokens_preds

    def cal_acc(self, logits, motion_tokens):
        logits_ = torch.nan_to_num(logits).detach()
        N, T_s, C = logits_.shape
        motion_probability = F.softmax(logits_, dim=-1) # N, Q, T, C
        motion_tokens_pred = motion_probability.argmax(dim=-1)
        
        # correct probability
        motion_probability = motion_probability.reshape((-1, C))
        motion_tokens_ = motion_tokens.reshape((-1))
        k_p = torch.stack([motion_probability[i, idx] for i, idx in enumerate(motion_tokens_)])
        correct_probability = torch.mean(k_p)
        
        # token acc
        tokens_acc = self.accuracy(motion_tokens_pred.reshape(-1), motion_tokens.reshape(-1))
        return motion_tokens_pred, correct_probability, tokens_acc
        
        

    def generate(self, batch: dict) -> torch.Tensor:
        """Run generate step on a batch of optional audio tensor and corresponding attributes.

       Args:
           batch (tuple[torch.Tensor, list[SegmentWithAttributes]]):
           use_prompt (bool): Whether to do audio continuation generation with prompt from audio batch.
           gen_duration (float): Target audio duration for the generation.
           prompt_duration (float, optional): Duration for the audio prompt to use for continuation.
           remove_prompt (bool, optional): Whether to remove the prompt from the generated audio.
           generation_params: Additional generation parameters.
       Returns:
           gen_outputs (dict): Generation outputs, consisting in audio, audio tokens from both the generation
               and the prompt along with additional information.
       """
        num_samples = None
        prompt_tokens = None
        attributes = None
        music = batch['audio']
        motion = batch['keypoints']
        self.motion_vqvae.eval()
        self.music_vqvae.eval()
        
        motion_tokens, _ = self.motion_vqvae.encode(motion)
        condition_tensors = self._prepare_tokens_and_attributes(music)
        if self.cfg.lm.lm_use:
            N, Q, T_s = motion_tokens.shape
            gen_tokens = self.model.generate(
                prompt_tokens=prompt_tokens, attributes=attributes, condition_tensors=condition_tensors, 
                max_gen_len=T_s,  num_samples=num_samples, **self.generation_params)
            # generate audio from tokens
            assert gen_tokens.dim() == 3
        else:
            logits = condition_tensors['wav'][0]
            motion_probability = F.softmax(logits, dim=-1) # N, Q, T, C
            gen_tokens = motion_probability.argmax(dim=-1)
            # generate audio from tokens
            assert gen_tokens.dim() == 2
        
        return gen_tokens
    
    def _compute_cross_entropy(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """Compute cross entropy between multi-codebook targets and model's logits.
        The cross entropy is computed per codebook to provide codebook-level cross entropy.
        Valid timesteps for each of the codebook are pulled from the mask, where invalid
        timesteps are set to 0.

        Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
        Returns:
            ce (torch.Tensor): Cross entropy averaged over the codebooks
            ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
        """
        B, K, T = targets.shape
        assert logits.shape[:-1] == targets.shape
        assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            ce_targets = targets_k[mask_k]
            ce_logits = logits_k[mask_k]
            q_ce = F.cross_entropy(ce_logits, ce_targets)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        # average cross entropy across codebooks
        ce = ce / K
        return ce, ce_per_codebook
    
    def register_ema(self, *args: str):
        """Register state sources for exponential moving average.

        The registered sources are used to instantiate a ModuleDictEMA instance.
        The ModuleDictEMA keeps a `nn.ModuleDict` module that is updated when self.ema.step() is called
        and swapped with the original state sources with self.swap_ema_state() method.

        Usage:
            self.register_ema('model')
        """
        assert self.ema is None, "Cannot register state source to already instantiated EMA."
        for name in args:
            self._ema_sources[name] = getattr(self, name)
    

    # def _generate_tokens(self, attributes: tp.List[ConditioningAttributes],
    #                      prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
    #     """Generate discrete audio tokens given audio prompt and/or conditions.

    #     Args:
    #         attributes (list of ConditioningAttributes): Conditions used for generation (text/melody).
    #         prompt_tokens (torch.Tensor, optional): Audio prompt used for continuation.
    #         progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
    #     Returns:
    #         torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
    #     """
    #     total_gen_len = int(self.duration * self.frame_rate)
    #     max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
    #     current_gen_offset: int = 0

    #     def _progress_callback(generated_tokens: int, tokens_to_generate: int):
    #         generated_tokens += current_gen_offset
    #         if self._progress_callback is not None:
    #             # Note that total_gen_len might be quite wrong depending on the
    #             # codebook pattern used, but with delay it is almost accurate.
    #             self._progress_callback(generated_tokens, tokens_to_generate)
    #         else:
    #             print(f'{generated_tokens: 6d} / {tokens_to_generate: 6d}', end='\r')

    #     if prompt_tokens is not None:
    #         assert max_prompt_len >= prompt_tokens.shape[-1], \
    #             "Prompt is longer than audio to generate"

    #     callback = None
    #     if progress:
    #         callback = _progress_callback

    #     if self.duration <= self.max_duration:
    #         # generate by sampling from LM, simple case.
    #         with self.autocast:
    #             gen_tokens = self.lm.generate(
    #                 prompt_tokens, attributes,
    #                 callback=callback, max_gen_len=total_gen_len, **self.generation_params)

    #     else:
    #         all_tokens = []
    #         if prompt_tokens is None:
    #             prompt_length = 0
    #         else:
    #             all_tokens.append(prompt_tokens)
    #             prompt_length = prompt_tokens.shape[-1]
            
    #         assert self.extend_stride is not None, "Stride should be defined to generate beyond max_duration"
    #         assert self.extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
    #         stride_tokens = int(self.frame_rate * self.extend_stride)
            
    #         while current_gen_offset + prompt_length < total_gen_len:
    #             time_offset = current_gen_offset / self.frame_rate
    #             chunk_duration = min(self.duration - time_offset, self.max_duration)
    #             max_gen_len = int(chunk_duration * self.frame_rate)
    #             with self.autocast:
    #                 gen_tokens = self.lm.generate(
    #                     prompt_tokens, attributes,
    #                     callback=callback, max_gen_len=max_gen_len, **self.generation_params)
    #             if prompt_tokens is None:
    #                 all_tokens.append(gen_tokens)
    #             else:
    #                 all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1]:])
    #             prompt_tokens = gen_tokens[:, :, stride_tokens:]
    #             prompt_length = prompt_tokens.shape[-1]
    #             current_gen_offset += stride_tokens

    #         gen_tokens = torch.cat(all_tokens, dim=-1)
    #     return gen_tokens
