import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .Encoder import VIT_GIN_Serial,VIT_GIN_Parallel
from .Decoder import TransformerDecoder
from einops import rearrange
from torch_geometric.data import Data
from thop import profile
from .utils import get_1d_sincos_pos_embed_from_grid,get_1d_sincos_pos_embed,sce_loss
class Pretrain_Serial(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=43, patch_size=1, in_chans=1, num_classes=2, embed_dim=108, depth=1,
                 num_heads=6, mlp_ratio=4.,qkv_bias=True, use_DropKey=True,drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 embed_layer=None, norm_layer=None, act_layer=None, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 norm_pix_loss=False):
        super().__init__()
        self.encoder = VIT_GIN_Serial(
            img_size=img_size,
            patch_size=patch_size, 
            in_chans=in_chans, 
            num_classes=num_classes, 
            embed_dim=embed_dim, 
            depth=depth, 
            num_heads=num_heads,
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias,
            use_DropKey=use_DropKey,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate
        )
        self.norm_pix_loss = norm_pix_loss
        self.decoder = TransformerDecoder(
            encoder = self.encoder,
            decoder_embed_dim = decoder_embed_dim,
            num_heads = decoder_num_heads, 
            depth = decoder_depth,
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate,   
            norm_layer=norm_layer, 
            act_layer=act_layer,
        )
    def forward_loss(self, imgs, pred_image, pred_graph, mask_image,mask_graph,loss_config=None):
        if loss_config is None:
            loss_config = {}
        target = self.encoder.vision.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            if target.shape[2] != 1:
                var = target.var(dim=-1, keepdim=True)
            else:
                var = 1
            target = (target - mean) / (var + 1.e-6).sqrt()
        loss1 = (pred_image - target) ** 2
        loss1 = loss1.mean(dim=-1)
        if mask_image.sum() != 0:
            loss1 = (loss1 * mask_image).sum() / mask_image.sum()
        else:
            loss1 = 0
        loss2 = sce_loss(pred_graph,target.squeeze(2),mask_graph)
        loss = (loss1+loss2)/2
        if 'orthogonal' in loss_config:
            # Orthogonal loss
            reg = loss_config['orthogonal']['reg'] 
            orth_loss = torch.zeros(1).to(imgs.device)
            for name, param in self.named_parameters():
                if 'bias' not in name:
                    param_flat = param.view(param.shape[0], -1)
                    sym = torch.mm(param_flat, torch.t(param_flat))
                    sym -= torch.eye(param_flat.shape[0]).to(param.device)
                    orth_loss += (reg * sym.abs().sum())
            loss += orth_loss.item()
        
        return loss

    def forward(self, data, mask_ratio_image=0.5, mask_ratio_graph=0.5,loss_config=None):
        x = data.x.view(-1,1,43,1)
        edge_index = data.edge_index
        latent_image, mask_image, ids_restore_image, latent_graph,mask_graph,ids_restore_graph = self.encoder.forward_features(x, edge_index, mask_ratio_image, mask_ratio_graph)
        pred_graph = self.decoder.forward_graph(latent_graph,edge_index,ids_restore_graph )
        pred_image = self.decoder.forward_image(latent_image, ids_restore_image)
        loss = self.forward_loss(x, pred_image, pred_graph, mask_image,mask_graph,loss_config)
        return loss
    def train_step(self,data,mask_ratio_image,mask_ratio_graph,optimizer,loss_config):
        # 模型训练步骤
        optimizer.zero_grad()  # 清除之前的梯度
        loss = self.forward(data,mask_ratio_image,mask_ratio_graph,loss_config)  # 前向传播
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        return loss.item() # 返回损失值
    def valid_step(self,data,mask_ratio_image,mask_ratio_graph,loss_config):
        loss = self.forward(data,mask_ratio_image,mask_ratio_graph,loss_config)  # 前向传播
        return loss.item()
    def test_step(self,data,mask_ratio_image,mask_ratio_graph,loss_config):
        loss = self.forward(data,mask_ratio_image,mask_ratio_graph,loss_config)
        return loss.item()
    def count_flops_and_params(self):
        #计算模型的FLOPs和参数量
        #:param input_size: 模型输入张量的尺寸
        #:return: FLOPs和参数量
        device = next(self.parameters()).device
        # 创建一个虚拟的Data对象用于性能分析
        # 注意，输入形状必须与你的模型输入匹配
        dummy_points = torch.randn(100, self.encoder.img_size).to(device)  # 根据模型输入调整大小
        dummy_adjs = torch.rand(100, 100).to(device)  # 根据模型输入调整大小
        rows, cols = torch.nonzero(dummy_adjs, as_tuple=True)
        edge_index = torch.stack([rows, cols]).to(device)
        dummy_data = Data(x=dummy_points, edge_index=edge_index)
        macs, params = profile(self, inputs=(dummy_data,), verbose=False)
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

class Pretrain_Parallel(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=43, patch_size=1, in_chans=1, num_classes=2, embed_dim=108, depth=1,
                 num_heads=6, mlp_ratio=4.,qkv_bias=True, use_DropKey=True,drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 embed_layer=None, norm_layer=None, act_layer=None, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 norm_pix_loss=False):
        super().__init__()
        self.encoder = VIT_GIN_Parallel(
            img_size=img_size,
            patch_size=patch_size, 
            in_chans=in_chans, 
            num_classes=num_classes, 
            embed_dim=embed_dim, 
            depth=depth, 
            num_heads=num_heads,
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias,
            use_DropKey=use_DropKey,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate
        )
        self.norm_pix_loss = norm_pix_loss
        self.decoder = TransformerDecoder(
            encoder = self.encoder,
            decoder_embed_dim = decoder_embed_dim,
            num_heads = decoder_num_heads, 
            depth = decoder_depth,
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate,   
            norm_layer=norm_layer, 
            act_layer=act_layer,
        )
    def forward_loss(self, imgs, pred_image, pred_graph, mask_image,mask_graph,loss_config=None):
        if loss_config is None:
            loss_config = {}
        target = self.encoder.vision.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            if target.shape[2] != 1:
                var = target.var(dim=-1, keepdim=True)
            else:
                var = 1
            target = (target - mean) / (var + 1.e-6).sqrt()
        loss1 = (pred_image - target) ** 2
        loss1 = loss1.mean(dim=-1)
        if mask_image.sum() != 0:
            loss1 = (loss1 * mask_image).sum() / mask_image.sum()
        else:
            loss1 = 0
        loss2 = sce_loss(pred_graph,target.squeeze(2),mask_graph)
        loss = (loss1+loss2)/2
        if 'orthogonal' in loss_config:
            # Orthogonal loss
            reg = loss_config['orthogonal']['reg'] 
            orth_loss = torch.zeros(1).to(imgs.device)
            for name, param in self.named_parameters():
                if 'bias' not in name:
                    param_flat = param.view(param.shape[0], -1)
                    sym = torch.mm(param_flat, torch.t(param_flat))
                    sym -= torch.eye(param_flat.shape[0]).to(param.device)
                    orth_loss += (reg * sym.abs().sum())
            loss += orth_loss.item()
        
        return loss

    def forward(self, data, mask_ratio_image=0.5, mask_ratio_graph=0.5,loss_config=None):
        x = data.x.view(-1,1,43,1)
        edge_index = data.edge_index
        latent_image, mask_image, ids_restore_image, latent_graph,mask_graph,ids_restore_graph = self.encoder.forward_features(x, edge_index, mask_ratio_image, mask_ratio_graph)
        pred_graph = self.decoder.forward_graph(latent_graph,edge_index,ids_restore_graph )
        pred_image = self.decoder.forward_image(latent_image, ids_restore_image)
        loss = self.forward_loss(x, pred_image, pred_graph, mask_image,mask_graph)
        return loss
    def train_step(self,data,mask_ratio_image,mask_ratio_graph,optimizer,loss_config):
        # 模型训练步骤
        optimizer.zero_grad()  # 清除之前的梯度
        loss = self.forward(data,mask_ratio_image,mask_ratio_graph,loss_config)  # 前向传播
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        return loss.item() # 返回损失值
    def valid_step(self,data,mask_ratio_image,mask_ratio_graph,loss_config):
        loss = self.forward(data,mask_ratio_image,mask_ratio_graph,loss_config)  # 前向传播
        return loss.item()
    def test_step(self,data,mask_ratio_image,mask_ratio_graph,loss_config):
        loss = self.forward(data,mask_ratio_image,mask_ratio_graph,loss_config)
        return loss.item()
    def count_flops_and_params(self):
        #计算模型的FLOPs和参数量
        #:param input_size: 模型输入张量的尺寸
        #:return: FLOPs和参数量
        device = next(self.parameters()).device
        # 创建一个虚拟的Data对象用于性能分析
        # 注意，输入形状必须与你的模型输入匹配
        dummy_points = torch.randn(100, self.encoder.img_size).to(device)  # 根据模型输入调整大小
        dummy_adjs = torch.rand(100, 100).to(device)  # 根据模型输入调整大小
        rows, cols = torch.nonzero(dummy_adjs, as_tuple=True)
        edge_index = torch.stack([rows, cols]).to(device)
        dummy_data = Data(x=dummy_points, edge_index=edge_index)
        macs, params = profile(self, inputs=(dummy_data,), verbose=False)
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