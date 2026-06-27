import torch
import torch.nn as nn


def _zero_init_linear(module):
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _empty_stats():
    return dict(
        stationary_gate_mean=0.0,
        stationary_gate_std=0.0,
        stationary_gate_q25=0.0,
        stationary_gate_q50=0.0,
        stationary_gate_q75=0.0,
        stationary_gate_scale=0.0,
        stationary_context_scale=0.0,
        stationary_safe_gate_mean=0.0,
        curve_gate_mean=0.0,
        curve_gate_std=0.0,
        curve_gate_q25=0.0,
        curve_gate_q50=0.0,
        curve_gate_q75=0.0,
        curve_context_scale=0.0,
        curve_msc_boost_scale=0.0,
    )


def _gate_stats(prefix, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32)
    if flat.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_q25": 0.0,
            f"{prefix}_q50": 0.0,
            f"{prefix}_q75": 0.0,
        }
    quantiles = torch.quantile(flat, torch.tensor([0.25, 0.5, 0.75], device=flat.device))
    return {
        f"{prefix}_mean": float(flat.mean().detach().cpu().item()),
        f"{prefix}_std": float(flat.std(unbiased=False).detach().cpu().item()),
        f"{prefix}_q25": float(quantiles[0].detach().cpu().item()),
        f"{prefix}_q50": float(quantiles[1].detach().cpu().item()),
        f"{prefix}_q75": float(quantiles[2].detach().cpu().item()),
    }


class _AuxControlBase(nn.Module):
    def __init__(self, dim=216, aux_dim=11, hidden_dim=64):
        super().__init__()
        self.aux_net = nn.Sequential(
            nn.LayerNorm(aux_dim),
            nn.Linear(aux_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.context_head = nn.Linear(hidden_dim, dim)
        _zero_init_linear(self.gate_head)
        _zero_init_linear(self.context_head)

    def _project(self, aux_features):
        hidden = self.aux_net(aux_features.to(torch.float32))
        gate = torch.sigmoid(self.gate_head(hidden))
        context = self.context_head(hidden)
        return gate, context


class StationaryDenseMSCControl(_AuxControlBase):
    def __init__(self, dim=216, aux_dim=11, hidden_dim=64):
        super().__init__(dim=dim, aux_dim=aux_dim, hidden_dim=hidden_dim)
        self.stationary_gate_scale_logit = nn.Parameter(torch.tensor([-4.0]))
        self.stationary_context_scale_logit = nn.Parameter(torch.tensor([-4.0]))

    def get_stationary_gate_scale(self):
        return 0.5 * torch.sigmoid(self.stationary_gate_scale_logit)

    def get_stationary_context_scale(self):
        return 0.3 * torch.sigmoid(self.stationary_context_scale_logit)

    def forward(self, aux_features):
        stationary_gate, stationary_context = self._project(aux_features)
        stationary_gate_scale = self.get_stationary_gate_scale()
        stationary_context_scale = self.get_stationary_context_scale()
        stationary_safe_gate = 1.0 - stationary_gate_scale * stationary_gate
        return dict(
            stationary_gate=stationary_gate,
            stationary_context=stationary_context,
            stationary_safe_gate=stationary_safe_gate,
            stationary_gate_scale=stationary_gate_scale,
            stationary_context_scale=stationary_context_scale,
        )


class RoadCurveMSCControl(_AuxControlBase):
    def __init__(self, dim=216, aux_dim=11, hidden_dim=64):
        super().__init__(dim=dim, aux_dim=aux_dim, hidden_dim=hidden_dim)
        self.curve_context_scale_logit = nn.Parameter(torch.tensor([-4.0]))
        self.curve_msc_boost_scale_logit = nn.Parameter(torch.tensor([-4.0]))

    def get_curve_context_scale(self):
        return 0.3 * torch.sigmoid(self.curve_context_scale_logit)

    def get_curve_msc_boost_scale(self):
        return 0.3 * torch.sigmoid(self.curve_msc_boost_scale_logit)

    def forward(self, aux_features):
        curve_gate, curve_context = self._project(aux_features)
        stationary_flag = aux_features[:, 7:8].to(curve_gate.device).to(torch.float32)
        curve_gate = curve_gate * (1.0 - stationary_flag.clamp(0.0, 1.0))
        return dict(
            curve_gate=curve_gate,
            curve_context=curve_context,
            curve_context_scale=self.get_curve_context_scale(),
            curve_msc_boost_scale=self.get_curve_msc_boost_scale(),
        )


class MSCControlFusion(nn.Module):
    def __init__(self, dim=216, aux_dim=11, mode="none", hidden_dim=64):
        super().__init__()
        if mode not in ("none", "sd", "rc", "rcsd"):
            raise ValueError(f"Unsupported msc_aux_mode: {mode}")
        self.mode = mode
        self.stationary_control = (
            StationaryDenseMSCControl(dim=dim, aux_dim=aux_dim, hidden_dim=hidden_dim)
            if mode in ("sd", "rcsd")
            else None
        )
        self.curve_control = (
            RoadCurveMSCControl(dim=dim, aux_dim=aux_dim, hidden_dim=hidden_dim)
            if mode in ("rc", "rcsd")
            else None
        )
        self._last_stats = _empty_stats()

    def reset_statistics(self):
        self._last_stats = _empty_stats()

    def get_statistics(self):
        return dict(self._last_stats)

    def forward(self, fused_feature, msc_context, segment_scale, aux_features):
        if aux_features is None:
            raise ValueError("AUX_FEATURES_REQUIRED")
        aux_features = aux_features.to(fused_feature.device).to(torch.float32)
        enhanced = fused_feature + segment_scale * msc_context
        stats = _empty_stats()

        if self.stationary_control is not None:
            stationary = self.stationary_control(aux_features)
            enhanced = (
                fused_feature
                + segment_scale * stationary["stationary_safe_gate"] * msc_context
                + stationary["stationary_context_scale"] * stationary["stationary_gate"] * stationary["stationary_context"]
            )
            stats.update(_gate_stats("stationary_gate", stationary["stationary_gate"]))
            stats["stationary_gate_scale"] = float(stationary["stationary_gate_scale"].detach().cpu().item())
            stats["stationary_context_scale"] = float(stationary["stationary_context_scale"].detach().cpu().item())
            stats["stationary_safe_gate_mean"] = float(
                stationary["stationary_safe_gate"].detach().mean().cpu().item()
            )

        if self.curve_control is not None:
            curve = self.curve_control(aux_features)
            enhanced = (
                enhanced
                + curve["curve_context_scale"] * curve["curve_gate"] * curve["curve_context"]
                + curve["curve_msc_boost_scale"] * curve["curve_gate"] * msc_context
            )
            stats.update(_gate_stats("curve_gate", curve["curve_gate"]))
            stats["curve_context_scale"] = float(curve["curve_context_scale"].detach().cpu().item())
            stats["curve_msc_boost_scale"] = float(curve["curve_msc_boost_scale"].detach().cpu().item())

        self._last_stats = stats
        return enhanced
