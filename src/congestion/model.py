"""LSTM-based multi-horizon traffic forecaster with an auxiliary congestion-level head."""
import torch
import torch.nn as nn


class CongestionLSTM(nn.Module):
    """
    Encodes `lookback` hours of (traffic volume + weather + calendar) features
    with a stacked LSTM, then predicts both:
      - raw traffic volume at each requested horizon (regression)
      - discrete congestion level {Low, Medium, High, Severe} at each horizon (classification)

    The two heads share the same recurrent encoder, which acts as a light
    regularizer (the classification signal keeps the representation aligned
    with decision-relevant congestion bands, not just curve-fitting volume).
    """

    def __init__(self, n_features, n_horizons, hidden_size=128, num_layers=2,
                 dropout=0.2, n_levels=4):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_levels = n_levels

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Attention-style pooling over time steps instead of just the last
        # hidden state, so the model can weigh e.g. a rush-hour spike 3
        # hours ago more heavily than the immediately preceding quiet hour.
        self.attn = nn.Linear(hidden_size, 1)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, n_horizons)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, n_horizons * n_levels)
        )

    def forward(self, x):
        # x: (batch, lookback, n_features)
        out, _ = self.lstm(x)                       # (batch, lookback, hidden)
        out = self.norm(out)
        weights = torch.softmax(self.attn(out), dim=1)  # (batch, lookback, 1)
        context = torch.sum(weights * out, dim=1)        # (batch, hidden)
        context = self.dropout(context)

        volume_pred = self.regressor(context)                                   # (batch, n_horizons)
        level_logits = self.classifier(context).view(-1, self.n_horizons, self.n_levels)
        return volume_pred, level_logits
