import torch
import torch.nn as nn


class DilatedTCNBlock(nn.Module):
    def __init__(self, dim, kernel_size=11, dilation=1, dropout=0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length")
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.block[3].weight)
        if self.block[3].bias is not None:
            nn.init.zeros_(self.block[3].bias)

    def forward(self, x):
        return x + self.block(x)


class MultiScaleSegmentContextModule(nn.Module):
    def __init__(self, dim=216, kernel_size=11, dilations=None, dropout=0.1):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32]
        self.norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            DilatedTCNBlock(dim, kernel_size=kernel_size, dilation=dilation, dropout=dropout)
            for dilation in dilations
        )
        self.segment_scale_logit = nn.Parameter(torch.tensor([-4.0]))
        self.reset_statistics()

    def reset_statistics(self):
        self._stat_count = 0
        self._fused_norm_before_sum = 0.0
        self._fused_norm_after_sum = 0.0
        self._context_norm_sum = 0.0
        self._context_to_fused_ratio_sum = 0.0

    def get_segment_scale(self):
        return 0.5 * torch.sigmoid(self.segment_scale_logit)

    def get_statistics(self):
        if self._stat_count == 0:
            return dict(
                segment_scale=float(self.get_segment_scale().detach().cpu().item()),
                fused_norm_mean_before_msc=0.0,
                fused_norm_mean_after_msc=0.0,
                context_norm_mean=0.0,
                context_to_fused_ratio=0.0,
            )
        return dict(
            segment_scale=float(self.get_segment_scale().detach().cpu().item()),
            fused_norm_mean_before_msc=self._fused_norm_before_sum / self._stat_count,
            fused_norm_mean_after_msc=self._fused_norm_after_sum / self._stat_count,
            context_norm_mean=self._context_norm_sum / self._stat_count,
            context_to_fused_ratio=self._context_to_fused_ratio_sum / self._stat_count,
        )

    def _accumulate_statistics(self, fused_before, fused_after, context):
        with torch.no_grad():
            before_norm = fused_before.norm(dim=1).mean()
            after_norm = fused_after.norm(dim=1).mean()
            context_norm = context.norm(dim=1).mean()
            ratio = context_norm / before_norm.clamp_min(1e-12)
            self._stat_count += 1
            self._fused_norm_before_sum += float(before_norm.detach().cpu().item())
            self._fused_norm_after_sum += float(after_norm.detach().cpu().item())
            self._context_norm_sum += float(context_norm.detach().cpu().item())
            self._context_to_fused_ratio_sum += float(ratio.detach().cpu().item())

    def forward(self, x):
        residual = x
        context = self.norm(x).transpose(0, 1).unsqueeze(0)
        for block in self.blocks:
            context = block(context)
        context = context.squeeze(0).transpose(0, 1)
        enhanced = residual + self.get_segment_scale() * context
        self._accumulate_statistics(residual, enhanced, context)
        return enhanced
