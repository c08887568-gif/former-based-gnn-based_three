import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange
from torch_geometric.data import Data
from .utils import label_smoothing_loss
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from thop import profile
from .GIN import GIN
from .VIT import VIT

class VIT_GIN_Parallel(nn.Module):
    def __init__(self, img_size=43, patch_size=1, in_chans=1, num_classes=2, embed_dim=108, depth=1,
                 num_heads=6, mlp_ratio=4.,qkv_bias=True,use_DropKey=True,
                 drop_rate=0., attn_drop_rate=0.0, drop_path_rate=0.0, embed_layer=None, norm_layer=None,
                 act_layer=None,pretrained_path=None):

        super().__init__()
        self.img_size = img_size
        self.vision = VIT(
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
            drop_path_rate=drop_path_rate, 
        )
        self.graph = GIN(img_size,embed_dim,0,drop_rate)
        self.head_image = nn.Linear(embed_dim,num_classes)
        self.image_norm = nn.LayerNorm(embed_dim)
        self.head_graph = nn.Linear(embed_dim,num_classes)
        self.head = nn.Linear(2*embed_dim,num_classes)
        if pretrained_path is not None:
            self.load_pretrained_encoder(pretrained_path)
    def forward_features(self, x, edge_index, mask_ratio_image=0,mask_ratio_graph=0):
        x_image,mask_image,ids_restore_image = self.vision.forward_features(x,mask_ratio_image)
        x_graph,mask_graph,ids_restore_graph = self.graph.forward_features(x.squeeze(),edge_index,mask_ratio_graph)
        return x_image,mask_image,ids_restore_image,x_graph,mask_graph,ids_restore_graph
    def forward(self, data):
        x = data.x.view(-1,1,43,1)
        edge_index = data.edge_index
        x_image,_,_,x_graph,_,_ = self.forward_features(x,edge_index)
        x_image = x_image[:, 1:, :].mean(dim=1) 
        x_image = self.image_norm(x_image)
        out_image = self.head_image(x_image)
        out_graph = self.head_graph(x_graph)
        out = torch.cat([x_image, x_graph], dim=1)
        out = self.head(out)
        return out,out_image,out_graph
    def forward_loss(self,pred,label,loss_config,pred_image=None,pred_graph=None):
        if loss_config is None:
            loss_config = {}
        epsilon = loss_config['label_smoothing']['epsilon'] if 'label_smoothing' in loss_config else 0
        loss = label_smoothing_loss(pred.float(), label, weight=torch.tensor([1.0, 1.0]).to(pred.device), epsilon=epsilon)
        if 'collaborative_training' in loss_config:
            tau = loss_config['collaborative_training']['tau']
            alpha = loss_config['collaborative_training']['alpha']
            pred_image *= tau
            pred_graph *= tau
            l1 = F.cross_entropy(pred_image, F.softmax(pred_graph, dim=1))
            l2 = F.cross_entropy(pred_graph, F.softmax(pred_image, dim=1))
            loss = alpha * loss + (1 - alpha) * (l1 + l2)
        if 'orthogonal' in loss_config:
            # Orthogonal loss
            reg = loss_config['orthogonal']['reg'] 
            orth_loss = torch.zeros((), device=pred.device)
            for name, param in self.named_parameters():
                if 'bias' not in name:
                    param_flat = param.view(param.shape[0], -1)
                    sym = torch.mm(param_flat, torch.t(param_flat))
                    sym -= torch.eye(param_flat.shape[0]).to(param.device)
                    orth_loss += (reg * sym.abs().sum())
            loss = loss + orth_loss
        return loss
    def train_step(self,data,label,optimizer,loss_config):
        # 模型训练步骤
        optimizer.zero_grad()  # 清除之前的梯度
        pred, pred_image, pred_graph = self.forward(data)  # 前向传播
        loss = self.forward_loss(pred, label,loss_config,pred_image,pred_graph)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        acc = (torch.argmax(pred, dim=1) == label.squeeze()).float().mean()
        return pred, loss.item() ,acc.item() # 返回损失值
    def valid_step(self,data,label,loss_config):
        pred, pred_image, pred_graph = self.forward(data)  # 前向传播
        loss = self.forward_loss(pred, label,loss_config,pred_image,pred_graph)  # 计算损失
        acc = (torch.argmax(pred, dim=1) == label.squeeze()).float().mean()
        return pred, loss.item() , acc.item()
    def test_step(self,data):
        pred, pred_image, pred_graph = self.forward(data)
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
        # 创建一个虚拟的Data对象用于性能分析
        # 注意，输入形状必须与你的模型输入匹配
        dummy_points = torch.randn(100, self.img_size).to(device)  # 根据模型输入调整大小
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
