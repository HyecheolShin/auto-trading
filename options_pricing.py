import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# GBM 경로 생성  [6_1], [10+11]_review, [12_1]
# ================================================================
def generate_brownian(n_paths, n_steps):
    dW = torch.randn(n_paths, n_steps)
    dW[:, 0] = 0.0
    return dW.cumsum(dim=-1)


def generate_geometric_brownian(n_paths=2, n_steps=250, sigma=0.2, dt=1/250):
    t = torch.arange(n_steps) * dt
    W = generate_brownian(n_paths, n_steps)
    return torch.exp((-0.5 * sigma**2) * t + sigma * torch.sqrt(torch.tensor(dt)) * W)


# ================================================================
# Payoff 함수  [12_1]_various_payoffs
# ================================================================
def european_payoff(spot, call=True, strike=1.0):
    if call:
        return F.relu(spot[..., -1] - strike)
    else:
        return F.relu(strike - spot[..., -1])


def lookback_payoff(spot, call=True, strike=1.0):
    if call:
        return F.relu(spot.max(dim=-1).values - strike)
    else:
        return F.relu(strike - spot.min(dim=-1).values)


def european_binary_payoff(spot, call=True, strike=1.0):
    if call:
        return (spot[..., -1] >= strike).to(spot)
    else:
        return (spot[..., -1] <= strike).to(spot)


def american_binary_payoff(spot, call=True, strike=1.0):
    if call:
        return (spot.max(dim=-1).values >= strike).to(spot)
    else:
        return (spot.min(dim=-1).values <= strike).to(spot)


# ================================================================
# 1. Black-Scholes  [8_1], [12_1]
# ================================================================
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes_delta(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


# ================================================================
# 2. Lookback (Monte Carlo)  [12_1]
# ================================================================
def simulate_gbm(S0, T, r, sigma, M, I):
    dt = T / M
    S = np.zeros((M + 1, I))
    S[0] = S0
    for t in range(1, M + 1):
        Z = np.random.standard_normal(I)
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return S


def lookback_option_monte_carlo(S0, K, T, r, sigma, M=50, I=10000,
                                 option_type="call", floating_strike=True):
    S = simulate_gbm(S0, T, r, sigma, M, I)
    if floating_strike:
        if option_type == "call":
            payoff = S[-1] - np.min(S, axis=0)
        else:
            payoff = np.max(S, axis=0) - S[-1]
    else:
        if option_type == "call":
            payoff = np.max(S, axis=0) - K
        else:
            payoff = K - np.min(S, axis=0)
    return np.exp(-r * T) * np.mean(np.maximum(payoff, 0))


# ================================================================
# 3. European Binary  [12_1]
# ================================================================
def european_binary_option(S, K, T, r, sigma, option_type="call"):
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return np.exp(-r * T) * norm.cdf(d2)
    else:
        return np.exp(-r * T) * norm.cdf(-d2)


# ================================================================
# 4. American Binary (Monte Carlo)  [12_1]
# ================================================================
def american_binary_option_monte_carlo(S0, K, T, r, sigma, M=50, I=10000,
                                        option_type="call", payout=1.0):
    S = simulate_gbm(S0, T, r, sigma, M, I)
    payoff = np.zeros(I)
    for i in range(I):
        if option_type == "call":
            if np.any(S[:, i] > K):
                payoff[i] = payout
        else:
            if np.any(S[:, i] < K):
                payoff[i] = payout
    return np.exp(-r * T) * np.mean(payoff)


# ================================================================
# P&L 계산  [9_2]_prev_hedge, [10+11]_review, [12_1]
# ================================================================
def pl(spot, unit, cost=None, payoff=None):
    output = unit[..., :-1].mul(spot.diff(dim=-1)).sum(dim=(-2, -1))
    if payoff is not None:
        output -= payoff.squeeze(-1)
    if cost is not None:
        c = torch.tensor(cost).unsqueeze(0).unsqueeze(-1)
        output -= (spot[..., :-1].mul(unit.diff(dim=-1)).abs() * c).sum(dim=(-2, -1))
    return output


# ================================================================
# 리스크 측도  [9_1_5]_entropic_risk_measure, [9_1], [10+11]_review, [12_1]
# ================================================================
def entropic_risk_measure(x, a=1.0):
    return (torch.logsumexp(-x * a, dim=0) - math.log(x.size(0))) / a


# ================================================================
# Feature 함수  [10+11]_review, [12_1]
# ================================================================
def time_to_maturity(spot, dt):
    n_paths, _, n_steps = spot.size()
    t = torch.arange(n_steps) * dt
    return (t[-1] - t).unsqueeze(0).expand(n_paths, 1, -1)


def moneyness(spot, strike):
    return spot / strike


def log_moneyness(spot, strike):
    return torch.log(spot / strike)


def volatility(spot, vol):
    return torch.ones_like(spot) * vol


# ================================================================
# Dataset  [8_1], [9_1], [10+11]_review, [12_1]
# ================================================================
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = torch.cat(data, dim=1)

    def __len__(self):
        return self.data.size(2)

    def __getitem__(self, index):
        return self.data[:, :, index].unsqueeze(1).to(device)


# ================================================================
# Deep Hedging 모델 (MLP + prev_hedge)  [9_2]_prev_hedge, [12_1]
# ================================================================
class MLP(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.register_buffer("prev_hedge", None)

    def forward(self, x):
        if self.prev_hedge is None:
            self.register_buffer("prev_hedge",
                                 torch.zeros(x.size(0), x.size(1), 1).to(device))
        new_x = torch.cat([x, self.prev_hedge], dim=-1)
        out = self.model(new_x)
        self.prev_hedge = out.detach()
        return out


# ================================================================
# 헤징 계산 유틸리티  [8_1], [9_1], [10+11]_review, [12_1]
# ================================================================
def compute_hedge(model, ds):
    outputs = []
    for i in ds:
        outputs.append(model(i))
    return torch.cat(outputs, dim=-1)


def compute_portfolio(model, ds, spot, payoff=None):
    unit = compute_hedge(model, ds)
    return pl(spot.to(device), unit, payoff=payoff)


# ================================================================
# Deep Hedging 학습  [9_1]_loss_function, [9_2]_prev_hedge
# ================================================================
def fit(model, ds, spot, payoff, n_epochs=200):
    optimizer = torch.optim.Adam(model.parameters())
    for i in range(n_epochs):
        optimizer.zero_grad()
        cash = compute_portfolio(model, ds, spot, payoff=payoff)
        loss = entropic_risk_measure(cash)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print(f"  epoch {i:3d}  loss={loss.item():.4f}")
    return model


# ================================================================
# 실행 예시
# ================================================================
if __name__ == "__main__":
    S0, K, T, r, sigma = 1.0, 1.1, 1.0, 0.0, 0.2
    np.random.seed(42)
    torch.manual_seed(42)

    spot = generate_geometric_brownian(n_paths=2000, n_steps=250).unsqueeze(1)

    print("=" * 55)
    print(f"S0={S0}, K={K}, T={T}yr, r={r}, sigma={sigma}")
    print("=" * 55)

    # 1. Black-Scholes
    print(f"\n[1] Black-Scholes (r=5%)")
    print(f"    Call: {black_scholes(S0, K, T, r=0.05, sigma=sigma, option_type='call'):.4f}")
    print(f"    Put : {black_scholes(S0, K, T, r=0.05, sigma=sigma, option_type='put'):.4f}")

    # 2. Lookback
    print(f"\n[2] Lookback (Monte Carlo)")
    print(f"    Call (floating): {lookback_option_monte_carlo(S0, K, T, r, sigma, option_type='call'):.4f}")
    print(f"    Put  (floating): {lookback_option_monte_carlo(S0, K, T, r, sigma, option_type='put'):.4f}")
    print(f"    Call (fixed K) : {lookback_option_monte_carlo(S0, K, T, r, sigma, option_type='call', floating_strike=False):.4f}")
    print(f"    Put  (fixed K) : {lookback_option_monte_carlo(S0, K, T, r, sigma, option_type='put',  floating_strike=False):.4f}")

    # 3. European Binary
    print(f"\n[3] European Binary (r=5%)")
    print(f"    Call: {european_binary_option(S0, K, T, r=0.05, sigma=sigma, option_type='call'):.4f}")
    print(f"    Put : {european_binary_option(S0, K, T, r=0.05, sigma=sigma, option_type='put'):.4f}")

    # 4. American Binary
    print(f"\n[4] American Binary (Monte Carlo)")
    print(f"    Call: {american_binary_option_monte_carlo(S0, K, T, r, sigma, option_type='call'):.4f}")
    print(f"    Put : {american_binary_option_monte_carlo(S0, K, T, r, sigma, option_type='put'):.4f}")

    # 5. Deep Hedging (MLP으로 American Binary 헤징)
    print(f"\n[5] Deep Hedging - American Binary (strike=1.1)")
    payoff = american_binary_payoff(spot, strike=1.1).to(device)

    dt = T / (spot.size(-1) - 1)
    lm  = log_moneyness(spot, K)
    ttm = time_to_maturity(spot, dt)
    vol = volatility(spot, sigma)
    ds  = MyDataset([lm, ttm, vol])

    model = MLP(n_inputs=4).to(device)
    fit(model, ds, spot, payoff, n_epochs=200)

    final_cash = compute_portfolio(model, ds, spot, payoff=payoff)
    print(f"  Entropic Risk (trained): {entropic_risk_measure(final_cash).item():.4f}")
    print("=" * 55)
