from typing import Optional
import torch

from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm
import torch.nn.functional as F
from torch_geometric.nn.inits import zeros



class SGConv(MessagePassing):
    r"""The simple graph convolutional operator from the `"Simplifying Graph
    Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        K (int, optional): Number of hops :math:`K`. (default: :obj:`1`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X}` on
            first execution, and will use the cached version for further
            executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:**
          node features :math:`(|\mathcal{V}|, F_{out})`
    """

    _cached_x: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int,is_attn, K: int = 3,
                 bias: bool = True, normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalize = normalize
        self.num_filters = 2  # 高频分量+低频分量

        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False) for _ in range(K + 1)
        ])
        self.filters = torch.nn.ModuleList([
            Linear(out_channels, out_channels, bias=bias) for _ in range(self.num_filters)
        ])
        self.weights = Linear(self.num_filters, 1, bias=bias)


        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.is_attn = is_attn

        #自注意力机制
        if self.is_attn:

            self.scale = out_channels ** -0.5
            self.qkv = torch.nn.Linear(in_channels, out_channels * 2, bias=False) #无偏置项
            self.attn_drop = torch.nn.Dropout(p=0.3)

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        if self.is_attn:

            res_out = self.lins[0](x)
            #拓扑半自适应图注意网络
            # 解码adj恢复方阵
            N, _ = x.shape
            _,C=res_out.shape
            adj = torch.zeros((N, N), dtype=torch.float32, device=x.device)
            adj[edge_index[0, :], edge_index[1, :]] = 1.0
            # 构建mask
            # 沿特征通道切割为不同头
            qkv = self.qkv(x).view(N, 2, 1, self.out_channels).permute(1, 2, 0, 3)
            q, k = qkv.unbind(0)
            attn = (torch.matmul(q, k.permute(0, 2, 1))) * self.scale
            attnclone = attn.clone()
            outputs = []
            # 构建semi adaptative masked attention adj

            # 多个头就是全局adj方阵与每个局部头的特征做mask，相当于一种另类的卷积
            mask_adj = -1e9 * (1.0 - adj)
            attnclone = attnclone + mask_adj
            attnclone = F.softmax(attnclone, dim=-1)
            attnclone = self.attn_drop(attnclone).squeeze()
            # 编码adj为原先格式
            edge_index_mask = torch.gt(attnclone, 0)
            rows, cols = torch.nonzero(edge_index_mask, as_tuple=True)
            edge_index_mask = torch.stack([rows, cols])
            if self.normalize:
                if isinstance(edge_index_mask, Tensor):
                    edge_index_mask, edge_weight = gcn_norm(  # yapf: disable
                        edge_index_mask, edge_weight, x.size(self.node_dim),
                        improved=False, add_self_loops=False, flow=self.flow,
                        dtype=x.dtype)

                elif isinstance(edge_index, SparseTensor):
                    edge_index_mask = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        add_self_loops=False, flow=self.flow, dtype=x.dtype)
            for i in range(self.num_filters):
                x_filtered=x
                res_out = self.lins[0](x) ##残差连接
                for lin in self.lins[1:]:
                    # propagate_type: (x: Tensor, edge_weight: OptTensor)
                    x_filtered = self.propagate(edge_index_mask, x=x_filtered, edge_weight=edge_weight)
                    sub_out = res_out + lin.forward(x_filtered) #残差连接
                    res_out=sub_out
                outputs.append(self.filters[i](sub_out))
            out = torch.mean(torch.stack(outputs), dim=0)
            if self.bias is not None:
                out = out + self.bias
        else:
            #这儿是原版TAGCN的实现
            if self.normalize:
                if isinstance(edge_index, Tensor):
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        improved=False, add_self_loops=False, flow=self.flow,
                        dtype=x.dtype)

                elif isinstance(edge_index, SparseTensor):
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        add_self_loops=False, flow=self.flow, dtype=x.dtype)

            out = self.lins[0](x)
            for lin in self.lins[1:]:
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
                out = out + lin.forward(x)

            if self.bias is not None:
                out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')
