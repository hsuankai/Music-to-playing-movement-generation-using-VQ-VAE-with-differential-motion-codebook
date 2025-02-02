import torch
import torch.nn as nn
import torch.nn.functional as F

# @torch.jit.script
def nonlinearity(x, act="swish"):
    # swish
    if act == "swish":
        return x*torch.sigmoid(x)
    elif act == "elu":
        return F.elu(x)
    elif act == "relu":
        return F.relu(x)

def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class Downsample1d(nn.Module):
    def __init__(self, in_channels, resamp_with_conv):
        super().__init__()
        self.with_conv = resamp_with_conv
        if resamp_with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            # TODO: can we replace it just with conv2d with padding 1?
            self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
            self.pad = (1, 1)
        else:
            self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
            
    def forward(self, x):
         if self.with_conv:  # bp: check self.avgpool and self.pad
             x = F.pad(x, self.pad, mode="constant", value=0)
             x = self.conv(x)
         else:
             x = self.avg_pool(x)
         return x

class Upsample1d(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            
    def forward(self, x, up_size=None):
        if up_size==None:
            x = F.interpolate(x, scale_factor=2.0, mode="linear")
        else:
            x = F.interpolate(x, size=up_size, mode="linear")
        if self.with_conv:
            x = self.conv(x)
        return x

class ResnetBlock1d(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, kernel_size=3, padding=1, act, num_groups):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.act = act
        self.num_groups = num_groups

        self.norm1 = Normalize(in_channels, num_groups)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels, num_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h, self.act)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb, self.act))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h, self.act)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h

class Encoder1d(nn.Module):
    def __init__(self, in_channels, out_ch, ch, ch_mult=(1,2,4), num_res_blocks=2,
                 dropout=0.0, resamp_with_conv=True, **ignore_kwargs):
        super().__init__()
        
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.act = ignore_kwargs['act']
        self.num_groups = ignore_kwargs['gn']
        
        # downsampling
        self.conv_in = torch.nn.Conv1d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock1d(in_channels=block_in,
                                           out_channels=block_out,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout, act=self.act, num_groups=self.num_groups))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample1d(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        if quant_conv:
            self.quant_conv = torch.nn.Conv1d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        else:
            self.quant_conv = torch.nn.Identity()

    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.quant_conv(h)
        return h
    
class Decoder1d(nn.Module):
    def __init__(self, *, in_channels, out_ch, ch, ch_mult=(1,2,4), num_res_blocks=2,
                 attn_resolutions=[53], dropout=0.0, resamp_with_conv=True, 
                 resolution=320, attn=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.attn = attn
        quant_conv = ignore_kwargs['quant_conv']
        self.act = ignore_kwargs['act']
        self.num_groups = ignore_kwargs['gn']

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        # self.z_shape = (1,z_channels,curr_res,curr_res)
        # print("Working with z of shape {} = {} dimensions.".format(
        #     self.z_shape, np.prod(self.z_shape)))

            
        self.conv_in = nn.Conv1d(in_channels, block_in, kernel_size=3, stride=1, padding=1)
        
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock1d(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout, act=self.act, num_groups=self.num_groups))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample1d(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in, self.num_groups)
        self.conv_out = torch.nn.Conv1d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
        
        

    def forward(self, z, final_len=None):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape
        h = self.conv_in(z)
        
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0 and i_level != 1:
                h = self.up[i_level].upsample(h)
            elif i_level == 1:
                h = self.up[i_level].upsample(h, up_size=final_len)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h, self.act)
        h = self.conv_out(h)
        return h
    
class Encoder1diff(nn.Module):
    def __init__(self, *, in_channels, out_ch, ch, ch_mult=(1,2,4), num_res_blocks=2,
                 attn_resolutions=[53], dropout=0.0, resamp_with_conv=True, 
                 resolution=320, attn=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.attn = attn
        quant_conv = ignore_kwargs['quant_conv']
        self.act = ignore_kwargs['act']
        self.num_groups = ignore_kwargs['gn']
        
        # downsampling
        self.conv_in = torch.nn.Conv1d(in_channels, self.ch, kernel_size=3,
                                       stride=1,
                                       padding=1)
        

        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock1d(in_channels=block_in,
                                           out_channels=block_out,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout, act=self.act, num_groups=self.num_groups))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample1d(block_in, resamp_with_conv)
            self.down.append(down)
        
        # end
        self.conv_out = nn.Conv1d(block_in, out_ch, kernel_size=1, stride=1, padding=0)
        self.quant_conv = nn.Conv1d(out_ch,  out_ch, kernel_size=1, stride=1, padding=0)
        self.quant_conv_diff = nn.Sequential(nn.Conv1d(out_ch, out_ch, kernel_size=1, stride=1, padding=0))
    
    def forward(self, x):
        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            if i_level != self.num_resolutions-1:
                h = self.down[i_level].downsample(h)
        
        h = self.conv_out(h)
        h_diff = h[..., 1:] - h[..., :-1] # hdiff1
        h = self.quant_conv(h)
        h_diff = self.quant_conv_diff(h_diff)
        return (h, h_diff)

