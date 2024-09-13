# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmseg.models.builder import BACKBONES
import math
from mmengine.runner import load_checkpoint
from functools import reduce
from operator import mul

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from mmseg.registry import MODELS


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        # x = self.dwconv(x, H, W)
        prompt_token = x[:, :10, :]
        x = self.dwconv(x[:, 10:, :], H, W)
        x = torch.cat([prompt_token, x], dim=1)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) # spatially reduction
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            prompt_token = x[:, :10, :]
            x_ = x[:, 10:, :].permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = torch.cat([prompt_token, x_], 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)


        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]

        self.num_patches = self.H * self.W

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W
    

class BlockwisePatchEmbedding(nn.Module):
    def __init__(
        self, num_channels, transformer_dim, patch_depth, patch_height, patch_width
    ):
        super().__init__()
        assert (
            num_channels % patch_depth == 0
        ), f"Number of channels {num_channels=} not divisible by patch_depth {patch_depth=}"
        self.patch_depth = patch_depth
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.transformer_dim = transformer_dim

        self.patch_dim = reduce(mul, [patch_depth, patch_height, patch_width])
        self.num_blocks = num_channels // patch_depth

        self.pre_norm = nn.LayerNorm(self.patch_dim)
        self.post_norm = nn.LayerNorm(self.transformer_dim)

        self.to_patch = Rearrange(
            "b (c p0) (h p1) (w p2) -> b c (h w) (p0 p1 p2)",
            p0=self.patch_depth,
            p1=self.patch_height,
            p2=self.patch_width,
        )
        self.blockwise_embed = nn.ModuleList(
            [
                nn.Linear(self.patch_dim, self.transformer_dim)
                for _ in range(self.num_blocks)
            ]
        )

    def embed(self, patches):
        patches = self.pre_norm(patches)

        embeds = []
        for i in range(self.num_blocks):
            embeds.append(self.blockwise_embed[i](patches[:, i, :, :]))

        embeds = torch.stack(embeds, dim=1)  # .flatten(start_dim=1, end_dim=2)
        embeds = rearrange(embeds, "b g n d -> b (g n) d")

        embeds = self.post_norm(embeds)

        return embeds

    def forward(self, x):
        patches = self.to_patch(x)

        embeddings = self.embed(patches)

        return embeddings


@MODELS.register_module()
class MixVisionTransformerVPT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], depth_spectral=[1,1,1,1], sr_ratios=[8, 4, 2, 1], freeze=True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        print(embed_dims)
        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # patch embed spectral
        self.patch_embed_spectral = BlockwisePatchEmbedding(32, embed_dims[0], 8, 8, 8)

        self.patch_embed_spectral2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed_spectral3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed_spectral4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])


        

        

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])


        #spectral encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1_spectral = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depth_spectral[0])])
        self.norm1_spectral = norm_layer(embed_dims[0])

        cur += depth_spectral[0]
        self.block2_spectral = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depth_spectral[1])])
        self.norm2_spectral = norm_layer(embed_dims[1])

        cur += depth_spectral[1]
        self.block3_spectral = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depth_spectral[2])])
        self.norm3_spectral = norm_layer(embed_dims[2])

        cur += depth_spectral[2]
        self.block4_spectral = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depth_spectral[3])])
        self.norm4_spectral = norm_layer(embed_dims[3])




        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        # vpt config
        self.token_depth = 0
        self.num_tokens_1 = 10
        self.num_tokens_2 = 10
        self.num_tokens_3 = 10
        self.num_tokens_4 = 10

        self.prompt_dropout = nn.Dropout(0)
        self.prompt_proj = nn.Identity()


        def prompt_init(prompt_embeddings, dim):
            val = math.sqrt(
                    6. / float(3 * reduce(mul, [patch_size, patch_size], 1) + dim))  # noqa
            nn.init.uniform_(prompt_embeddings.data, -val, val)

        self.prompt_cfg = kwargs['prompt_cfg']
        if 'deep' in self.prompt_cfg:
            self.prompt_embeddings_1 = nn.Parameter(torch.zeros(self.depths[0], self.num_tokens_1, self.embed_dims[0]))
            prompt_init(self.prompt_embeddings_1, self.embed_dims[0])
            self.prompt_embeddings_2 = nn.Parameter(torch.zeros(self.depths[1], self.num_tokens_2, self.embed_dims[1]))
            prompt_init(self.prompt_embeddings_2, self.embed_dims[1])
            self.prompt_embeddings_3 = nn.Parameter(torch.zeros(self.depths[2], self.num_tokens_3, self.embed_dims[2]))
            prompt_init(self.prompt_embeddings_3, self.embed_dims[2])
            self.prompt_embeddings_4 = nn.Parameter(torch.zeros(self.depths[3], self.num_tokens_4, self.embed_dims[3]))
            prompt_init(self.prompt_embeddings_4, self.embed_dims[3])
        else:
            self.prompt_embeddings_1 = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dims[0]))
            prompt_init(self.prompt_embeddings_1, self.embed_dims[0])

        if freeze:
            self.freeze()

        model_total_params = sum(p.numel() for p in self.parameters())
        model_grad_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params),
              '\nmodel_total_params:' + str(model_total_params))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                

    def init_weights(self, pretrained='./segformer.b3.512x512.ade.160k.pth'):
        print("pesos", pretrained)
        if isinstance(pretrained, str):
            print("pesos cargados")
            
            # Load the checkpoint
            checkpoint = torch.load(pretrained, map_location='cpu')
            
            # Create a new state dict with modified keys
            new_state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('backbone.'):
                    new_k = k.replace('backbone.', '', 1)
                else:
                    new_k = k
                new_state_dict[new_k] = v
            
            # Load the modified state dict
            self.load_state_dict(new_state_dict, strict=False)


    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]


    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better


    def get_classifier(self):
        return self.head


    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x, multimodal=False):
        B = x.shape[0]
        outs = []

        if multimodal:
            spectral = x[:, 2:, :, :]
            x = x[:, :3, :, :]
            x_spectral = self.patch_embed_spectral(spectral)

        # -------
        # stage 1
        # -------
        x, H, W = self.patch_embed1(x) 

        for i, blk in enumerate(self.block1):

            if multimodal:
                
                if  i+1 <= len(self.block1_spectral):
                    x_spectral = torch.cat([ self.prompt_dropout(self.prompt_proj(self.prompt_embeddings_1[i])).expand(B, -1, -1), x_spectral], dim=1)
                
                    x_spectral = self.block1_spectral[i](x_spectral, H + self.token_depth, W)


                    spectral_prompts = x_spectral[:, :self.num_tokens_1, :] # remove spectral tokens and save prompts
                    x = torch.cat([spectral_prompts, x], dim=1) # add spectral prompts to RGB image

                    x_spectral = x_spectral[:, self.num_tokens_1:, :] # remove prompts
                else:
                    x = torch.cat([self.prompt_dropout(self.prompt_proj(self.prompt_embeddings_1[i])).expand(B, -1, -1), x],dim=1)
                    
            else:

                x = torch.cat([self.prompt_dropout(self.prompt_proj(self.prompt_embeddings_1[i])).expand(B, -1, -1), x],dim=1)

            x = blk(x, H + self.token_depth, W)

            x = x[:, self.num_tokens_1:, :] # remove prompts

        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if multimodal:
            x_spectral = self.norm1_spectral(x_spectral)
            x_spectral = x_spectral.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        outs.append(x)

        # -------
        # stage 2
        # -------

        x, H, W = self.patch_embed2(x)

        if multimodal:
            x_spectral, _, _ = self.patch_embed_spectral2(x_spectral)


        for i, blk in enumerate(self.block2):

            if multimodal:
                
                if  i+1 <= len(self.block2_spectral):
                    x_spectral = torch.cat([ self.prompt_dropout(self.prompt_proj(self.prompt_embeddings_2[i])).expand(B, -1, -1), x_spectral], dim=1)
                
                    x_spectral = self.block2_spectral[i](x_spectral, H + self.token_depth, W)


                    spectral_prompts = x_spectral[:, :self.num_tokens_2, :] # remove spectral tokens and save prompts
                    x = torch.cat([spectral_prompts, x], dim=1) # add spectral prompts to RGB image

                    x_spectral = x_spectral[:, self.num_tokens_2:, :] # remove prompts
                else:
                    x = torch.cat([self.prompt_dropout(self.prompt_proj(self.prompt_embeddings_2[i])).expand(B, -1, -1), x],dim=1)
                    
            else:

                x = torch.cat([self.prompt_dropout(self.prompt_proj(self.prompt_embeddings_2[i])).expand(B, -1, -1), x],dim=1)

            x = blk(x, H + self.token_depth, W)

            x = x[:, self.num_tokens_2:, :] # remove prompts


        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        if multimodal:
            x_spectral = self.norm2_spectral(x_spectral)
            x_spectral = x_spectral.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        outs.append(x)

        # -------
        # stage 3
        # -------
        x, H, W = self.patch_embed3(x)

        if multimodal:
            x_spectral, _, _ = self.patch_embed_spectral3(x_spectral)

        for i, blk in enumerate(self.block3):

            if multimodal:
                
                if  i+1 <= len(self.block3_spectral):
                    x_spectral = torch.cat([ self.prompt_dropout(self.prompt_proj(self.prompt_embeddings_3[i])).expand(B, -1, -1), x_spectral], dim=1)
                
                    x_spectral = self.block3_spectral[i](x_spectral, H + self.token_depth, W)


                    spectral_prompts = x_spectral[:, :self.num_tokens_3, :] # remove spectral tokens and save prompts
                    x = torch.cat([spectral_prompts, x], dim=1) # add spectral prompts to RGB image

                    x_spectral = x_spectral[:, self.num_tokens_3:, :] # remove prompts
                else:
                    x = torch.cat([self.prompt_dropout(self.prompt_proj(self.prompt_embeddings_3[i])).expand(B, -1, -1), x],dim=1)
                    
            else:

                x = torch.cat([self.prompt_dropout(self.prompt_proj(self.prompt_embeddings_3[i])).expand(B, -1, -1), x],dim=1)

            x = blk(x, H + self.token_depth, W)

            x = x[:, self.num_tokens_3:, :] # remove prompts

        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if multimodal:
            x_spectral = self.norm3_spectral(x_spectral)
            x_spectral = x_spectral.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        outs.append(x)

        # -------
        # stage 4
        # -------
        x, H, W = self.patch_embed4(x)

        if multimodal:
            x_spectral, _ , _ = self.patch_embed_spectral4(x_spectral)

        for i, blk in enumerate(self.block4):

            if multimodal:
                
                if  i+1 <= len(self.block4_spectral):
                    x_spectral = torch.cat([ self.prompt_dropout(self.prompt_proj(self.prompt_embeddings_4[i])).expand(B, -1, -1), x_spectral], dim=1)
                
                    x_spectral = self.block4_spectral[i](x_spectral, H + self.token_depth, W)

                    spectral_prompts = x_spectral[:, :self.num_tokens_4, :] # remove spectral tokens and save prompts
                    x = torch.cat([spectral_prompts, x], dim=1) # add spectral prompts to RGB image

                    x_spectral = x_spectral[:, self.num_tokens_4:, :] # remove prompts
                else:
                    x = torch.cat([self.prompt_dropout(self.prompt_proj(self.prompt_embeddings_4[i])).expand(B, -1, -1), x],dim=1)
                    
            else:

                x = torch.cat([self.prompt_dropout(self.prompt_proj(self.prompt_embeddings_4[i])).expand(B, -1, -1), x],dim=1)

            x = blk(x, H + self.token_depth, W)

            x = x[:, self.num_tokens_4:, :] # remove prompts

        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        outs.append(x)

        return outs


    def forward(self, x):

        if x.shape[1] > 3:
            x = self.forward_features(x, multimodal=True)
        else:
            print("forward missing modality!")
            x = self.forward_features(x, multimodal=False)

        # x = self.head(x)

        return x


    def freeze(self):
        for k, p in self.named_parameters():
            if "prompt" not in k and "decode_head" not in k and "spectral" not in k:
                p.requires_grad = False


@BACKBONES.register_module()
class mit_b0_vpt(MixVisionTransformerVPT):
    def __init__(self, **kwargs):
        super(mit_b0_vpt, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


@BACKBONES.register_module()
class mit_b1_vpt(MixVisionTransformerVPT):
    def __init__(self, **kwargs):
        super(mit_b1_vpt, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


@BACKBONES.register_module()
class mit_b2_vpt(MixVisionTransformerVPT):
    def __init__(self, **kwargs):
        super(mit_b2_vpt, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


@BACKBONES.register_module()
class mit_b3_vpt(MixVisionTransformerVPT):
    def __init__(self, **kwargs):
        super(mit_b3_vpt, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


@BACKBONES.register_module()
class mit_b4_vpt(MixVisionTransformerVPT):
    def __init__(self, **kwargs):
        super(mit_b4_vpt, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


@BACKBONES.register_module()
class mit_b5_vpt(MixVisionTransformerVPT):
    def __init__(self, **kwargs):
        super(mit_b5_vpt, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)