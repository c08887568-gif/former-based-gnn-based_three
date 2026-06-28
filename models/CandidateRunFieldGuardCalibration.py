import torch
import torch.nn as nn


CANDCAL_RUN_GUARD_DEFAULTS = {
    "hidden_dim": 64,
    "dropout": 0.1,
    "max_delta": 0.45,
    "threshold_candidates": [0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    "min_run_len": 5,
    "positive_field_ratio": 0.80,
    "negative_field_ratio": 0.20,
    "pos_weight_clip": 10.0,
}


class CandidateRunFieldGuardCalibration(nn.Module):
    def __init__(self, input_dim, config=None):
        super().__init__()
        self.config = dict(CANDCAL_RUN_GUARD_DEFAULTS)
        if config:
            self.config.update(config)
        hidden_dim = int(self.config["hidden_dim"])
        dropout = float(self.config["dropout"])
        self.max_delta = float(self.config["max_delta"])
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, run_features):
        return self.net(run_features).squeeze(-1)

    def score(self, run_features):
        return torch.sigmoid(self.forward(run_features))

    def calibration_delta(self, run_features):
        return self.max_delta * self.score(run_features)


def compute_pos_weight(targets, pos_weight_clip=None):
    targets = torch.as_tensor(targets, dtype=torch.float32)
    positives = int((targets >= 0.5).sum().item())
    negatives = int((targets < 0.5).sum().item())
    if positives == 0 or negatives == 0:
        return None, positives, negatives
    pos_weight = float(negatives) / max(float(positives), 1.0)
    clip = CANDCAL_RUN_GUARD_DEFAULTS["pos_weight_clip"] if pos_weight_clip is None else pos_weight_clip
    pos_weight = min(pos_weight, float(clip))
    return torch.tensor(pos_weight, dtype=torch.float32), positives, negatives
