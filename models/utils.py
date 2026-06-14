import torch
import torch.nn.functional as F
def label_smoothing_loss(pred, label, weight, epsilon=0.1):
    n_class = pred.shape[1]
    one_hot = torch.nn.functional.one_hot(label.view(-1), n_class).float()
    smoothed_one_hot = (1.0 - epsilon) * one_hot + epsilon / n_class
    loss = torch.nn.functional.cross_entropy(pred.float(), smoothed_one_hot, weight=weight, reduction='mean')
    return loss
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / (10000**omega)

    pos = pos.reshape(-1)
    out = torch.einsum('m,d->md', pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb
def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid = torch.arange(grid_size, dtype=torch.float32).reshape(1, grid_size)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed

# 定义 SCE 损失函数

def sce_loss(x, y, mask,alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = (loss*mask).sum()/mask.sum()
    return loss