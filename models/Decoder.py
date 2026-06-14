import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .utils import get_1d_sincos_pos_embed_from_grid,get_1d_sincos_pos_embed
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
 
 
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
 
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# MLP module
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# Patch Embed module
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_size, patch_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else Identity()

    def forward(self, x):
        B, C, H, W = x.size()
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BLD
        x = self.norm(x)
        return x

# Attention module
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,attn_drop=0., proj_drop=0., use_DropKey=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.use_DropKey = use_DropKey
        self.mask_ratio = attn_drop
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.use_DropKey == True:
            m_r = torch.ones_like(attn)*self.mask_ratio
            attn = attn + torch.bernoulli(m_r)*-1e12
        attn = F.softmax(attn, dim=-1)
        if self.use_DropKey != True:
            attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Block module
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,use_DropKey=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class TransformerDecoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self,
                 encoder=None,
                 decoder_embed_dim = 512,
                 num_heads = 8, 
                 depth = 1,
                 mlp_ratio=4.0, 
                 qkv_bias=True, 
                 drop_rate=0.0, 
                 attn_drop_rate=0.0, 
                 drop_path_rate=0.0,  
                 norm_layer=None, 
                 act_layer=None,
                ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_patches = encoder.vision.patch_embed.num_patches
        self.tra_mask_token_decoder=nn.Parameter(nn.init.trunc_normal_(torch.zeros(1,decoder_embed_dim),std=0.02))
        self.decoder_embed = nn.Linear(encoder.vision.embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(nn.init.trunc_normal_(torch.zeros(1, 1, decoder_embed_dim),std=0.02))
        self.decoder_pos_embed = nn.Parameter(nn.init.trunc_normal_(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),std=0.02))
        self.decoder_pos_embed.requires_grad = True
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
            for _ in range(depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, encoder.vision.patch_size**2 * encoder.vision.in_chans)
        self.tradecoder_embed = nn.Linear(encoder.graph.embed_dim, decoder_embed_dim)
        self.tracls_token = nn.Parameter(nn.init.trunc_normal_(torch.zeros(1, 1, decoder_embed_dim),std=0.02))
        self.tradecoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
            for _ in range(depth)])
        self.tradecoder_norm = norm_layer(decoder_embed_dim)
        self.tradecoder_pred = nn.Linear(decoder_embed_dim, encoder.vision.img_size * encoder.vision.in_chans)
        self.initialize_weights()
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}
    def initialize_weights(self):
        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches+1, cls_token=False)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.clone().float().unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            #nn.init.trunc_normal_(m.weight)
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_image(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = x_[torch.arange(x.shape[0])[:, None], ids_restore]
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_graph(self, x,adj,ids_restore):
        x = self.tradecoder_embed(x)
        mask_tokens = self.tra_mask_token_decoder.repeat(ids_restore.shape[0] + 1 - x.shape[0], 1)
        x = torch.cat([x, mask_tokens], dim=0)
        x = x[ids_restore]
        x=x.unsqueeze(0)
        cls_tokens = self.tracls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        for blk in self.tradecoder_blocks:
            x = blk(x)
        x = self.tradecoder_norm(x)
        x = self.tradecoder_pred(x)
        x = x[:, 1:, :]
        return x