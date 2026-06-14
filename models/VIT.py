import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange
from .utils import label_smoothing_loss
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from thop import profile
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
class VIT(nn.Module):
    def __init__(self, img_size=43, patch_size=1, in_chans=1, num_classes=2, embed_dim=108, depth=1,
                 num_heads=6, mlp_ratio=4., qkv_bias=True,use_DropKey=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=None, norm_layer=None,
                 act_layer=None,pretrained_path=None):

        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim 
        norm_layer = norm_layer or nn.LayerNorm
        act_layer = act_layer or nn.GELU
        embed_layer = embed_layer or PatchEmbed
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,use_DropKey=use_DropKey, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.fc_norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim,num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        if pretrained_path is not None:
            self.load_pretrained_encoder(pretrained_path)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # elif isinstance(m, nn.Conv2d):
        #     nn.init.trunc_normal_(m.weight, std=0.2)
        #     if isinstance(m, nn.Conv2d) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
    def patchify(self, imgs):
        p = self.patch_embed.proj.kernel_size[0]
        assert imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.view(imgs.shape[0], self.in_chans, h, p, w, p)
        x = rearrange(x, 'n c h p w q -> n (h w) (p q c)')
        return x
    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_shuffle[:,:len_keep] =torch.sort(ids_shuffle[:, :len_keep], dim=1)[0]
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = x[torch.arange(N)[:, None], ids_keep]

        mask = torch.ones(N, L)
        mask[:, :len_keep] = 0
        mask = mask[torch.arange(N)[:, None], ids_restore]
        return x_masked, mask.to(x.device), ids_restore.to(x.device)
    
    def forward_features(self, x, mask_ratio=0):
        B = x.size(0)
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return x,mask,ids_restore
    def forward(self, x):
        x,_,_ = self.forward_features(x)
        x = x[:, 1:, :].mean(dim=1) 
        x = self.fc_norm(x)
        out = self.head(x)
        return out
    def forward_loss(self,pred,label,loss_config):
        if loss_config is None:
            loss_config = {}
        epsilon = loss_config['label_smoothing']['epsilon'] if 'label_smoothing' in loss_config else 0
        loss = label_smoothing_loss(pred.float(), label, weight=torch.tensor([1.0, 1.0]).to(pred.device), epsilon=epsilon)
        if 'orthogonal' in loss_config:
            # Orthogonal loss
            reg = loss_config['orthogonal']['reg'] 
            orth_loss = torch.zeros(1).to(pred.device)
            for name, param in self.named_parameters():
                if 'bias' not in name:
                    param_flat = param.view(param.shape[0], -1)
                    sym = torch.mm(param_flat, torch.t(param_flat))
                    sym -= torch.eye(param_flat.shape[0]).to(param.device)
                    orth_loss += (reg * sym.abs().sum())
            loss += orth_loss.item()
        return loss
    def train_step(self,x,label,optimizer,loss_config):
        # 模型训练步骤
        optimizer.zero_grad()  # 清除之前的梯度
        pred = self.forward(x)  # 前向传播
        loss = self.forward_loss(pred, label,loss_config)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        acc = (torch.argmax(pred, dim=1) == label.squeeze()).float().mean()
        return pred, loss.item() ,acc.item() # 返回损失值
    def valid_step(self,x,label,loss_config):
        pred = self.forward(x)  # 前向传播
        loss = self.forward_loss(pred, label,loss_config)  # 计算损失
        acc = (torch.argmax(pred, dim=1) == label.squeeze()).float().mean()
        return pred, loss.item() , acc.item()
    def test_step(self,x):
        pred = self.forward(x)
        return pred
    def calculate_classification_metrics(self,pred,label):
        prelabel = np.argmax(pred, axis=1)
        class_result = classification_report(label, prelabel, digits=4)
        return class_result
    def load_pretrained_encoder(self, pretrained_path):
        encoder_state_dict = torch.load(pretrained_path)

        # Filter and remove 'encoder.' prefix from keys
        new_encoder_state_dict = OrderedDict()
        for k, v in encoder_state_dict.items():
            if k.startswith('encoder.'):
                new_key = k.replace('encoder.', '')
                new_encoder_state_dict[new_key] = v

        # Get the model's current state dict
        model_state_dict = self.state_dict()

        # Update the model state dict with the new encoder parameters
        model_state_dict.update(new_encoder_state_dict)
        self.load_state_dict(model_state_dict, strict=False)
    def count_flops_and_params(self):
        #计算模型的FLOPs和参数量
        #:param input_size: 模型输入张量的尺寸
        #:return: FLOPs和参数量
        device = next(self.parameters()).device
        # 创建一个虚拟的x对象用于性能分析
        # 注意，输入形状必须与你的模型输入匹配
        dummy_images = torch.randn(100,1, self.img_size,1).to(device)  # 根据模型输入调整大小
        macs, params = profile(self, inputs=(dummy_images,), verbose=False)
        self.clear_thop_hooks_and_attributes()
        return macs, params
    def clear_thop_hooks_and_attributes(self):
        """
        清除 THOP 添加的所有钩子和自定义属性。
        """
        def remove_thop_hooks_and_attributes(module):
            if hasattr(module, "_forward_pre_hooks"):
                module._forward_pre_hooks.clear()
            if hasattr(module, "_forward_hooks"):
                module._forward_hooks.clear()
            if hasattr(module, "_backward_hooks"):
                module._backward_hooks.clear()

            # 清除自定义属性
            if hasattr(module, "total_ops"):
                delattr(module, "total_ops")
            if hasattr(module, "total_params"):
                delattr(module, "total_params")

            # 递归清理子模块
            for name, child in module.named_children():
                remove_thop_hooks_and_attributes(child)

        self.apply(remove_thop_hooks_and_attributes)

