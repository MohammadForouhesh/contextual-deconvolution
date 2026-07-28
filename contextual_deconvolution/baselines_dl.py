"""
Deep learning baselines: N-BEATS, PatchTST, DeepAR.
PyTorch implementations for demand forecasting.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting
# ---------------------------------------------------------------------------

class NBeatsBlock(nn.Module):
    """Generic N-BEATS block with basis expansion."""
    def __init__(self, input_size: int, output_size: int, hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        layers = []
        for i in range(n_layers):
            in_dim = input_size if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.fc_stack = nn.Sequential(*layers)
        self.basis = nn.Linear(hidden_dim, output_size)
        self.backcast = nn.Linear(hidden_dim, input_size)

    def forward(self, x):
        # x: (batch, input_size)
        h = self.fc_stack(x)
        forecast = self.basis(h)
        backcast = self.backcast(h)
        return backcast, forecast


class NBeatsStack(nn.Module):
    """Stack of N-BEATS blocks with residual connections."""
    def __init__(self, input_size: int, output_size: int, n_blocks: int = 3, hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, output_size, hidden_dim, n_layers)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        # x: (batch, input_size)
        forecast = torch.zeros(x.size(0), self.blocks[0].output_size, device=x.device)
        for block in self.blocks:
            backcast, block_forecast = block(x)
            x = x - backcast  # residual connection
            forecast = forecast + block_forecast
        return forecast


class NBeatsModel(nn.Module):
    """Full N-BEATS with trend and seasonality stacks (simplified)."""
    def __init__(self, input_size: int, output_size: int, hidden_dim: int = 128, n_blocks: int = 3, n_layers: int = 4):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.stack = NBeatsStack(input_size, output_size, n_blocks, hidden_dim, n_layers)

    def forward(self, x):
        return self.stack(x)


# ---------------------------------------------------------------------------
# PatchTST: Long-term Forecasting with Transformers
# ---------------------------------------------------------------------------

class PatchTST(nn.Module):
    """
    Simplified PatchTST: patch time series, project, apply transformer attention, forecast.
    """
    def __init__(self, input_size: int, output_size: int, patch_len: int = 16, stride: int = 8,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.patch_len = patch_len
        self.stride = stride
        # Number of patches
        self.n_patches = max(1, (input_size - patch_len) // stride + 1)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_patches * d_model, output_size)
        )

    def forward(self, x):
        # x: (batch, input_size)
        batch = x.size(0)
        # Create patches
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # (batch, n_patches, patch_len)
        # Embed patches
        patches = self.patch_embedding(patches)  # (batch, n_patches, d_model)
        # Transformer
        enc = self.transformer(patches)  # (batch, n_patches, d_model)
        # Forecast head
        out = self.head(enc)  # (batch, output_size)
        return out


# ---------------------------------------------------------------------------
# DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
# ---------------------------------------------------------------------------

class DeepAR(nn.Module):
    """
    Simplified DeepAR: LSTM encoder + Gaussian output.
    """
    def __init__(self, input_size: int, hidden_dim: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0
        )
        self.mu_proj = nn.Linear(hidden_dim, 1)
        self.sigma_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, 1)
        out, hidden = self.lstm(x, hidden)
        mu = self.mu_proj(out)  # (batch, seq_len, 1)
        sigma = F.softplus(self.sigma_proj(out)) + 1e-3  # (batch, seq_len, 1)
        return mu, sigma, hidden


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def create_sequences(data: np.ndarray, lookback: int, horizon: int):
    """Create sliding window sequences."""
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon])
    return np.array(X), np.array(y)


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, lr=1e-3, patience=7, device='cpu'):
    """Generic training loop with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        optimizer.zero_grad()
        pred = model(X_t)
        loss = F.mse_loss(pred, y_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
            y_v = torch.tensor(y_val, dtype=torch.float32, device=device)
            val_pred = model(X_v)
            val_loss = F.mse_loss(val_pred, y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Public interface matching existing baselines.py
# ---------------------------------------------------------------------------

def run_nbeats(b: np.ndarray, f_promo: np.ndarray, f_snap: np.ndarray,
               forecast_horizon: int = 28, lookback: int = 56, device: str = 'cpu'):
    """Train N-BEATS and return forecasts."""
    try:
        T = len(b)
        if T < lookback + forecast_horizon + 50:
            lookback = min(lookback, T // 3)
        split = T - forecast_horizon
        b_train = b[:split]

        # Add features as extra channels (simplified: concat to demand)
        features = np.stack([b, f_promo, f_snap], axis=1)  # (T, 3)
        X, y = create_sequences(features[:, 0], lookback, forecast_horizon)
        if len(X) < 20:
            return None, None, None, None

        # Train/val split
        n_train = int(0.8 * len(X))
        X_tr, y_tr = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:], y[n_train:]

        model = NBeatsModel(input_size=lookback, output_size=forecast_horizon, hidden_dim=128, n_blocks=3, n_layers=4)
        model = train_model(model, X_tr, y_tr, X_val, y_val, epochs=100, lr=1e-3, patience=15, device=device)

        # Forecast: use last lookback days
        model.eval()
        with torch.no_grad():
            x_input = torch.tensor(b_train[-lookback:].reshape(1, -1), dtype=torch.float32, device=device)
            pred = model(x_input).cpu().numpy().flatten()

        actual = b[split:]
        fitted = np.concatenate([b_train[:lookback], model(torch.tensor(X, dtype=torch.float32, device=device)).detach().cpu().numpy()[:, 0]])
        return pred, actual, fitted, None
    except Exception as e:
        import warnings
        warnings.warn(f"N-BEATS failed: {e}")
        return None, None, None, None


def run_patchtst(b: np.ndarray, f_promo: np.ndarray, f_snap: np.ndarray,
                 forecast_horizon: int = 28, lookback: int = 56, device: str = 'cpu'):
    """Train PatchTST and return forecasts."""
    try:
        T = len(b)
        if T < lookback + forecast_horizon + 50:
            lookback = min(lookback, T // 3)
        split = T - forecast_horizon
        b_train = b[:split]

        X, y = create_sequences(b_train, lookback, forecast_horizon)
        if len(X) < 20:
            return None, None, None, None

        n_train = int(0.8 * len(X))
        X_tr, y_tr = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:], y[n_train:]

        patch_len = min(16, lookback // 4)
        stride = max(1, patch_len // 2)
        model = PatchTST(input_size=lookback, output_size=forecast_horizon, patch_len=patch_len, stride=stride,
                         d_model=64, n_heads=4, n_layers=2)
        model = train_model(model, X_tr, y_tr, X_val, y_val, epochs=100, lr=1e-3, patience=15, device=device)

        model.eval()
        with torch.no_grad():
            x_input = torch.tensor(b_train[-lookback:].reshape(1, -1), dtype=torch.float32, device=device)
            pred = model(x_input).cpu().numpy().flatten()

        actual = b[split:]
        fitted = np.concatenate([b_train[:lookback], model(torch.tensor(X, dtype=torch.float32, device=device)).detach().cpu().numpy()[:, 0]])
        return pred, actual, fitted, None
    except Exception as e:
        import warnings
        warnings.warn(f"PatchTST failed: {e}")
        return None, None, None, None


def train_deepar(model, X_train, y_train, X_val, y_val, epochs=50, lr=1e-3, patience=7, device='cpu'):
    """Training loop for DeepAR with Gaussian NLL.
    X_train: (n_samples, seq_len) input sequences
    y_train: (n_samples, seq_len) target sequences (same as X for autoregressive)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device).unsqueeze(-1)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(-1)
    X_v = torch.tensor(X_val, dtype=torch.float32, device=device).unsqueeze(-1)
    y_v = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(-1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        mu, sigma, _ = model(X_t)
        loss = torch.mean(torch.log(sigma) + 0.5 * ((y_t - mu) / sigma) ** 2)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            mu_v, sigma_v, _ = model(X_v)
            val_loss = torch.mean(torch.log(sigma_v) + 0.5 * ((y_v - mu_v) / sigma_v) ** 2).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_deepar(b: np.ndarray, f_promo: np.ndarray, f_snap: np.ndarray,
               forecast_horizon: int = 28, lookback: int = 56, device: str = 'cpu', n_samples: int = 100):
    """Train DeepAR and return probabilistic forecasts (mean of samples)."""
    try:
        T = len(b)
        if T < lookback + forecast_horizon + 50:
            lookback = min(lookback, T // 3)
        split = T - forecast_horizon
        b_train = b[:split]

        # For DeepAR, X and y are the same sequence (autoregressive prediction)
        X, y = create_sequences(b_train, lookback, lookback)
        if len(X) < 20:
            return None, None, None, None

        n_train = int(0.8 * len(X))
        X_tr, y_tr = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:], y[n_train:]

        model = DeepAR(input_size=lookback, hidden_dim=64, n_layers=2)
        model = train_deepar(model, X_tr, y_tr, X_val, y_val, epochs=50, lr=1e-3, patience=7, device=device)

        model.eval()
        with torch.no_grad():
            x_input = torch.tensor(b_train[-lookback:].reshape(1, -1), dtype=torch.float32, device=device).unsqueeze(-1)
            # Autoregressive sampling for forecast horizon
            samples = []
            for _ in range(n_samples):
                curr = x_input.clone()
                hidden = None
                preds = []
                for _ in range(forecast_horizon):
                    mu, sigma, hidden = model(curr, hidden)
                    mu_t = mu[:, -1, 0]
                    sigma_t = sigma[:, -1, 0]
                    z = torch.randn_like(mu_t) * sigma_t + mu_t
                    preds.append(z.item())
                    # Append to sequence
                    z_in = z.unsqueeze(1).unsqueeze(2)
                    curr = torch.cat([curr[:, 1:, :], z_in], dim=1)
                samples.append(preds)
            pred = np.mean(samples, axis=0)

        actual = b[split:]
        # Fitted: one-step predictions on training data
        with torch.no_grad():
            X_all = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(-1)
            mu_all, _, _ = model(X_all)
            fitted = np.concatenate([b_train[:lookback], mu_all[:, -1, 0].detach().cpu().numpy()])
        return pred, actual, fitted, None
    except Exception as e:
        import warnings
        warnings.warn(f"DeepAR failed: {e}")
        return None, None, None, None
