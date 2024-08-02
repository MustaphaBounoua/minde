# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0,
                                                 end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=1)

        x = x + \
            gate_msa.unsqueeze(
                1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + \
            gate_mlp.unsqueeze(
                1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x





class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_size,nb_var,encoding=False):
        super().__init__()
        self.encoding = encoding
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, out_size, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        if self.encoding==True:
            #return self.linear(x.view(x.shape[0],-1))
            return self.linear(x) #.sum(dim=1)
        else:
            return self.linear(x)
     


class VarDeEmbed(nn.Module):
    """ Mod to mod Embedding
    """

    def __init__(
            self,
            sizes=[],
            embed_dim: int = 768,
            norm_layer=None,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = sizes
        self.sizes = sizes

        norm_l = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.proj = nn.ModuleList(
            [nn.Linear(embed_dim, size, bias=bias) for size in sizes])
        self.norm = nn.ModuleList([norm_l for size in sizes])

    def forward(self, x_var, i=[]):
        x_var = x_var.permute(1, 0, 2)
        if len(i) == 0:
            i = np.arange(len(x_var))
        proj = np.array(self.proj)[i]
        x = [
            proj_x(x_var[idx]) for idx, proj_x in enumerate(proj)
        ]

        return x


class VarEmbed(nn.Module):
    """ Var to Var Embedding
    """

    def __init__(
            self,
            sizes=[],
            embed_dim: int = 768,
            norm_layer=None,
            bias: bool = True,
            variable_input=False,
    ):
        super().__init__()

        self.sizes = sizes
        self.variable_input = variable_input
        norm_l = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.norm = nn.ModuleList([norm_l for size in sizes])
        self.proj = nn.ModuleList(
            [nn.Linear(size, embed_dim, bias=bias) for size in sizes])

    def forward(self, x_var):
        
        i = np.arange(len(x_var))

        proj = np.array(self.proj)[i]
        norm = np.array(self.norm)[i]

        x = [
            proj_x(x_var[idx].float()) for idx, proj_x in enumerate(proj)
        ]

        x = [
            norm_x(x[idx]) for idx, norm_x in enumerate(norm)
        ]
        return torch.stack(x).permute(1, 0, 2)




class DiT_Enc(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        hidden_size=1152,
        var_sizes=[],
        depth=2,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.var_sizes = var_sizes
        self.hidden_size = hidden_size

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, hidden_size,nb_var=len(var_sizes),encoding=True)
        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t):

        c = t
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        hidden_size=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        var_list=None,
        type = "c"
    ):
        super().__init__()
        self.num_heads = num_heads
        self.var_sizes = list(var_list.values())
        self.var_list = var_list
        self.type = type
        self.hidden_size = hidden_size
        if type =="c":
            self.var_enc = DiT_Enc(hidden_size=hidden_size,
                                var_sizes= [self.var_sizes[1]],
                                depth=depth//2,
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratio,
                                )
        else:
            self.var_enc = DiT_Enc(hidden_size=hidden_size,
                                var_sizes=self.var_sizes,
                                depth=depth//2,
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratio,
                                )

        self.x_embedder = VarEmbed(sizes=self.var_sizes,
                                   embed_dim=hidden_size,
                                   norm_layer=nn.LayerNorm)
        
        
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, hidden_size,encoding=False,nb_var=len(self.var_sizes))
        # if type =="c":
        #     self.unembed_var = VarDeEmbed(sizes=[self.var_sizes[0]],
        #                               embed_dim=hidden_size,
        #                               norm_layer=None)
        # else:
        self.unembed_var = VarDeEmbed(sizes=self.var_sizes,
                                      embed_dim=hidden_size,
                                      norm_layer=None)
        #self.initialize_weights()
        self.pos_embed = self.get_pos_embed(np.array(np.arange(len(self.var_sizes))))

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        for mod in self.x_embedder.proj:
            w = mod.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(mod.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        for var in self.unembed_var.proj:
            w = var.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(var.bias, 0)
            
        

    def get_pos_embed(self, i):
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, i)
        return torch.from_numpy(pos_embed).float().unsqueeze(0)

    def forward(self, x, t=None, mask=None,std=None):

        x = torch.split(x, self.var_sizes, dim=1)
                           

        
        x_all = self.x_embedder(x) + self.pos_embed.to(x[0].device)  
        
        
        t = self.t_embedder(t).squeeze()
        
        
        

        if self.type =="c":
            x = x_all[:,0,:].unsqueeze(1)
            if mask[0][1]==0:
                mask_cond = ((mask == 0).sum(dim=1) > 0).int().view(mask.shape[0], 1)
                y = self.var_enc(x_all[:,1,:].unsqueeze(1), t=torch.zeros_like(t))
                y = y.sum(dim=1)
            else:
                y = torch.zeros_like(t)
        elif self.type =="j"  : 
            mask = mask.view(mask.shape[0], mask.shape[1], 1)
            x, x_c = (mask > 0).int() * x_all, (mask == 0).int() * x_all
            mask_cond = ((mask == 0).sum(dim=1) > 0).int().view(mask.shape[0], 1)
            y = self.var_enc(x_c, t=torch.zeros_like(t))
            y = y.sum(dim=1) *mask_cond
 
        c = t + y
        
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unembed_var(x)
        if self.type =="c":
            x.append(torch.zeros_like(x[0]))
        out = torch.cat(x, dim=1)
    
        if std != None:
            return out/std
        else:
            return out


  