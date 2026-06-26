import torch
import torch.nn as nn
import torch.nn.functional as F
# from mysg_conv3 import SGConv
from .mysg_conv3 import SGConv, SGConvEdgeTypeWeightPretrain, SGConvEdgeWeightPretrain
from torch_geometric.nn import global_mean_pool
from collections import OrderedDict
from einops import rearrange
from torch_geometric.data import Data
from .utils import label_smoothing_loss
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from thop import profile
import numpy as np
class GIN(nn.Module):
    def __init__(self, num_features, embed_dim,num_classes,drop_rate,pretrained_path=None,
                 pretrain_mode="current", num_edge_types=4):
        super(GIN, self).__init__()
        self.embed_dim = embed_dim
        self.num_features = num_features
        self.pretrain_mode = pretrain_mode
        self.num_edge_types = num_edge_types
        # self.conv1 = GINConv(nn.Sequential(
        #     nn.Linear(num_features, embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.ReLU()
        # ))
        # self.conv2 = GINConv(nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.ReLU()
        # ))

        # self.conv1 = SGConv(num_features,embed_dim,is_adaptation=True)
        # self.conv2 = SGConv(embed_dim,embed_dim,is_adaptation=False)
        if pretrain_mode == "edge_weight":
            self.conv1 = SGConvEdgeWeightPretrain(num_features,embed_dim,is_attn=False)
            self.conv2 = SGConvEdgeWeightPretrain(embed_dim,embed_dim,is_attn=False)
        elif pretrain_mode == "edge_type_weight":
            self.conv1 = SGConvEdgeTypeWeightPretrain(num_features,embed_dim,is_attn=False,num_edge_types=num_edge_types)
            self.conv2 = SGConvEdgeTypeWeightPretrain(embed_dim,embed_dim,is_attn=False,num_edge_types=num_edge_types)
        else:
            self.conv1 = SGConv(num_features,embed_dim,is_attn=True)
            self.conv2 = SGConv(embed_dim,embed_dim,is_attn=False)
        # self.conv1 = SGConv(num_features,embed_dim)
        # self.conv2 = SGConv(embed_dim,embed_dim)
        self.drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        self.reset_pretrain_edge_learners()
        if pretrained_path is not None:
            self.load_pretrained_encoder(pretrained_path)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def reset_pretrain_edge_learners(self):
        for conv in (self.conv1, self.conv2):
            learner = getattr(conv, "edge_weight_learner", None)
            if learner is not None and hasattr(learner, "reset_parameters"):
                learner.reset_parameters()
            learner = getattr(conv, "edge_type_weight_learner", None)
            if learner is not None and hasattr(learner, "reset_parameters"):
                learner.reset_parameters()
    def graph_random_masking(self, x, edge_index,mask_ratio, edge_weight=None):
        L, D = x.shape
        if mask_ratio <= 0:
            mask = torch.zeros(L, device=x.device)
            ids_restore = torch.arange(L, device=x.device)
            return x, edge_index, edge_weight, mask, ids_restore

        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(L, device=x.device)
        ids_shuffle = torch.argsort(noise)
        ids_shuffle[:len_keep] = torch.sort(ids_shuffle[:len_keep])[0]
        ids_restore = torch.argsort(ids_shuffle)

        ids_keep = ids_shuffle[:len_keep]
        x_masked = x[ids_keep]
        adj = torch.zeros((L, L), dtype=torch.float32, device=x.device)
        adj[edge_index[0, :], edge_index[1,:]] = 1.0
        adj_weight = None
        if edge_weight is not None:
            adj_weight = torch.zeros((L, L), dtype=torch.float32, device=x.device)
            adj_weight[edge_index[0, :], edge_index[1, :]] = edge_weight.to(x.device).to(torch.float32)
        adj_masked = adj[:,ids_keep][ids_keep,:]
        rows, cols = torch.nonzero(adj_masked, as_tuple=True)
        # 按照源节点和目标节点的顺序构建新的张量
        edge_index_masked = torch.stack([rows, cols]).to(x.device)
        edge_weight_masked = None
        if adj_weight is not None:
            edge_weight_masked = adj_weight[:, ids_keep][ids_keep, :][rows, cols]
        mask = torch.ones(L, device=x.device)
        mask[:len_keep] = 0
        mask = mask[ ids_restore]
        return x_masked, edge_index_masked, edge_weight_masked, mask, ids_restore
    def forward_features(self, x, edge_index,mask_ratio=0, edge_weight=None):
        x, edge_index, edge_weight, mask,ids_restore  = self.graph_random_masking(x,edge_index ,mask_ratio, edge_weight)
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = self.drop(x)
        return x,mask,ids_restore
    def get_pretrain_edge_statistics(self):
        weights = []
        probs = []
        for conv in (self.conv1, self.conv2):
            edge_weight = getattr(conv, "last_edge_weight", None)
            edge_type_prob = getattr(conv, "last_edge_type_prob", None)
            if edge_weight is not None:
                weights.append(edge_weight.reshape(-1).detach().cpu())
            if edge_type_prob is not None:
                probs.append(edge_type_prob.detach().cpu())
        result = {}
        if weights:
            edge_weight = torch.cat(weights)
            q25, q50, q75 = torch.quantile(edge_weight, torch.tensor([0.25, 0.5, 0.75]))
            result["edge_weight"] = dict(
                mean=float(edge_weight.mean().item()),
                std=float(edge_weight.std(unbiased=False).item()),
                min=float(edge_weight.min().item()),
                q25=float(q25.item()),
                q50=float(q50.item()),
                q75=float(q75.item()),
                max=float(edge_weight.max().item()),
            )
        if probs:
            edge_type_prob = torch.cat(probs, dim=0)
            mean_prob = edge_type_prob.mean(dim=0)
            entropy = -(edge_type_prob * torch.log(edge_type_prob.clamp_min(1e-12))).sum(dim=1)
            dominant = edge_type_prob.argmax(dim=1)
            dominant_ratio = max((dominant == idx).float().mean().item() for idx in range(edge_type_prob.shape[1]))
            result["edge_type"] = dict(
                mean_prob=[float(value.item()) for value in mean_prob],
                entropy_mean=float(entropy.mean().item()),
                dominant_type_ratio=float(dominant_ratio),
                type_collapse_flag=bool(dominant_ratio > 0.95),
            )
        return result
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        x,_,_ = self.forward_features(x,edge_index,edge_weight=edge_weight)
        out = self.head(x)
        return F.log_softmax(out, dim=1)
    def forward_loss(self,pred,label,loss_config):
        if loss_config is None:
            loss_config = {}
        epsilon = loss_config['label_smoothing']['epsilon'] if 'label_smoothing' in loss_config else 0
        loss = label_smoothing_loss(pred.float(), label, weight=torch.tensor([1.0, 1.0]).to(pred.device), epsilon=epsilon)
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
        pred = self.forward(data)  # 前向传播
        loss = self.forward_loss(pred, label,loss_config)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        acc = (torch.argmax(pred, dim=1) == label.squeeze()).float().mean()
        return pred, loss.item() ,acc.item() # 返回损失值
    def valid_step(self,data,label,loss_config):
        pred = self.forward(data)  # 前向传播
        loss = self.forward_loss(pred, label,loss_config)  # 计算损失
        acc = (torch.argmax(pred, dim=1) == label.squeeze()).float().mean()
        return pred, loss.item() , acc.item()
    def test_step(self,data):
        pred = self.forward(data)
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
        dummy_points = torch.randn(100, self.num_features).to(device)  # 根据模型输入调整大小
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
