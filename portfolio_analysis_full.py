
# ===============================
# Portfolio Analysis – Full Script
# ===============================
# Requirements:
#   - Python 3.9+
#   - pip install pandas numpy scipy matplotlib openpyxl
#
# Input:
#   - A file named 'returns.xlsx' (Sheet1), with a 'DATE' column and the following columns:
#       'Australian Listed Equity [G]',
#       "Int'l Listed Equity (Hedged) [G]",
#       "Int'l Listed Equity (Unhedged) [G]",
#       'Australian Listed Property [G]',
#       "Int'l Listed Property [G]",
#       "Int'l Listed Infrastructure [G]",
#       'Australian Fixed Income [D]',
#       "Int'l Fixed Income (Hedged) [D]",
#       'Cash [D]'
#
# What it does:
#   1) Loads data and computes monthly log-returns.
#   2) Splits into Period A (2012-01 to 2015-12) and Period B (2016-01 to 2019-12).
#   3) Computes mean vectors and covariance matrices on log-returns.
#   4) Converts to multi-period (1y, 2y) expected arithmetic returns/covariances/correlations under a lognormal mapping.
#   5) Computes unconstrained mean-variance optimal portfolio (closed form) for a risk aversion parameter t.
#   6) Computes risk-free + risky portfolio weights (tangency-like using Cash as risk-free asset).
#   7) Solves constrained optimization (SLSQP) with practical constraints (min allocations, growth bands, weights>=0, sum=1).
#   8) Evaluates realized 2y out-of-sample returns for 2020–2021 and 2021–2022 using those optimized weights.
#
# Notes:
#   - The script prints key results. You can tailor constraints easily in build_constraints().
#   - This script assumes monthly data and 12 months per year when scaling.
# ===============================

import pandas as pd
import numpy as np
from scipy.optimize import minimize

# -------------------------------
# Configuration
# -------------------------------
EXCEL_PATH = "returns.xlsx"
SHEET_NAME = "Sheet1"
INDEX_COL = "Unnamed: 0"

ASSETS = [
    'Australian Listed Equity [G]',
    "Int'l Listed Equity (Hedged) [G]",
    "Int'l Listed Equity (Unhedged) [G]",
    'Australian Listed Property [G]',
    "Int'l Listed Property [G]",
    "Int'l Listed Infrastructure [G]",
    'Australian Fixed Income [D]',
    "Int'l Fixed Income (Hedged) [D]",
    'Cash [D]',
]

PERIOD_A = ("2012-01-01", "2015-12-31")
PERIOD_B = ("2016-01-01", "2019-12-31")
OOOS = [("2020-01-01", "2021-12-31"),
        ("2021-01-01", "2022-12-31")]  # Out-of-sample windows

np.set_printoptions(suppress=True, linewidth=160)

# -------------------------------
# Utilities
# -------------------------------
def safe_read_excel(path: str, sheet: str, index_col: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, sheet_name=sheet, index_col=index_col)
        return df.copy()
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] File '{path}' not found. Please place it next to this script.")
    except Exception as e:
        raise SystemExit(f"[ERROR] Failed to read Excel: {e}")

def compute_log_returns(df: pd.DataFrame, asset_cols: list) -> pd.DataFrame:
    out = df.copy()
    for col in asset_cols:
        out[f'log_{col}'] = np.log(1 + out[col])
    return out

def slice_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df['DATE'] >= start) & (df['DATE'] <= end)].copy()

def mean_cov_from_logs(df_log: pd.DataFrame, asset_cols: list):
    cols = [f'log_{c}' for c in asset_cols]
    X = df_log[cols].copy()
    a = X.mean().values           # mean vector of log-returns
    B = X.cov().values            # covariance matrix of log-returns
    return a, B

def multi_period_moments_from_log(a, B, k_years: int, m_per_year: int = 12):
    n = k_years * m_per_year  # number of months
    mu = n * a                # log mean aggregated
    Sigma = n * B             # log covariance aggregated
    sigma2 = np.diag(Sigma)   # variances

    # Expected arithmetic returns for each asset over n months under lognormal:
    exp_R = np.exp(mu + 0.5 * sigma2) - 1.0

    # Variances for each asset:
    var_i = np.exp(2 * mu + 2 * sigma2) - np.exp(2 * mu + sigma2)

    # Full covariance matrix for arithmetic returns (lognormal mapping)
    E = np.exp(mu + 0.5 * sigma2)
    outer_E = np.outer(E, E)
    Cov = outer_E * (np.exp(Sigma) - 1.0)
    np.fill_diagonal(Cov, var_i)

    # Correlation
    std = np.sqrt(np.diag(Cov))
    Corr = Cov / np.outer(std, std)
    return exp_R, Cov, Corr

def closed_form_opt_weights(r, C, t=1.0):
    """
    Unconstrained mean-variance optimal weights for risky assets.
    Returns w = alpha + t * beta, where alpha is the minimum-variance portfolio
    and beta shifts toward higher expected return.
    """
    e = np.ones(len(r))
    Cinv = np.linalg.inv(C)
    a = e @ Cinv @ e
    b = r @ Cinv @ e

    alpha = (Cinv @ e) / a
    beta = Cinv @ r - (b / a) * (Cinv @ e)
    w = alpha + t * beta
    return w, dict(a=a, b=b, alpha=alpha, beta=beta)

def tangency_with_cash(risky_exp_ret, risky_cov, cash_return, t=1.0):
    """
    Risk-free + risky weights using Cash as risk-free return.
    For t=1:
      cash_w = 1 - 1' * Cinv * (r - Rf * 1)
      risky_w = Cinv * (r - Rf * 1)
    (This is shown for documentation only; actual code uses vectors/matrices.)
    """
    e = np.ones(len(risky_exp_ret))
    Cinv = np.linalg.inv(risky_cov)
    r_bar = risky_exp_ret - e * cash_return
    cash_w = 1 - (e @ Cinv @ r_bar) * t
    risky_w = (Cinv @ r_bar) * t
    return cash_w, risky_w

def objective_MV(w, mu, cov):
    # Standard mean-variance scalarization: minimize -mu'w + 0.5 w' C w
    return -float(w @ mu) + 0.5 * float(w @ cov @ w)

def build_constraints(num_assets, growth_slice=slice(0, 6), min_constraints=True):
    cons = [ {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0} ]
    if min_constraints:
        # Example minimums (tweak as needed):
        cons += [
            {'type': 'ineq', 'fun': lambda w: w[0] - 0.15},  # AUS Eq >= 15%
            {'type': 'ineq', 'fun': lambda w: w[1] - 0.15},  # Intl Eq (H) >= 15%
            {'type': 'ineq', 'fun': lambda w: w[6] - 0.05},  # AUS FI >= 5%
            {'type': 'ineq', 'fun': lambda w: w[8] - 0.05},  # Cash >= 5%
        ]
        # Optional: growth assets band between 65% and 75% (indexes 0..5)
        cons += [
            {'type': 'ineq', 'fun': lambda w: np.sum(w[growth_slice]) - 0.65},
            {'type': 'ineq', 'fun': lambda w: 0.75 - np.sum(w[growth_slice])},
        ]
    return cons

def solve_constrained_mv(mu, cov, bounds=None, constraints=None, w0=None):
    n = len(mu)
    if bounds is None:
        bounds = [(0.0, 1.0)] * n
    if constraints is None:
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    if w0 is None:
        w0 = np.ones(n) / n
    res = minimize(objective_MV, w0, args=(mu, cov), method='SLSQP',
                   bounds=bounds, constraints=constraints)
    return res

def pretty_weights(weights, labels):
    return pd.DataFrame({'Asset': labels, 'Weight': np.round(weights, 6)})

def realized_cumulative_returns(df_slice, asset_cols):
    # Converts monthly simple returns to cumulative over the slice: prod(1+r) - 1
    df_ret = df_slice[asset_cols] + 1.0
    cum = df_ret.prod() - 1.0
    return cum.values

def main():
    print("=== Loading data ===")
    df = safe_read_excel(EXCEL_PATH, SHEET_NAME, INDEX_COL)
    if 'DATE' not in df.columns:
        raise SystemExit("[ERROR] 'DATE' column not found.")
    df['DATE'] = pd.to_datetime(df['DATE'])

    # log-returns
    df_log = compute_log_returns(df, ASSETS)

    # Split periods
    df_A = slice_period(df_log, *PERIOD_A)
    df_B = slice_period(df_log, *PERIOD_B)

    # Mean/cov on logs
    a_A, B_A = mean_cov_from_logs(df_A, ASSETS)
    a_B, B_B = mean_cov_from_logs(df_B, ASSETS)

    # Print
    print("\nPeriod A mean (log, monthly):")
    print(pd.Series(np.round(a_A, 6), index=ASSETS))
    print("\nPeriod A cov (log, monthly):")
    print(np.round(B_A, 6))

    print("\nPeriod B mean (log, monthly):")
    print(pd.Series(np.round(a_B, 6), index=ASSETS))
    print("\nPeriod B cov (log, monthly):")
    print(np.round(B_B, 6))

    # Multi-period stats (1y and 2y) from logs -> arithmetic
    stats = {}
    for label, (a, B) in {'A': (a_A, B_A), 'B': (a_B, B_B)}.items():
        for k in [1, 2]:
            ER, COV, CORR = multi_period_moments_from_log(a, B, k_years=k, m_per_year=12)
            stats[(label, k)] = dict(ER=ER, COV=COV, CORR=CORR)

            print(f"\n--- Period {label}, Return {k} year(s) ---")
            print("Expected Returns (arithmetic):")
            print(pd.Series(np.round(ER, 6), index=ASSETS))
            print("\nCovariance matrix (arithmetic):")
            print(np.round(COV, 6))
            print("\nCorrelation matrix:")
            print(np.round(CORR, 4))

    # Choose 2-year stats for optimization examples
    mu_A = stats[('A', 2)]['ER']
    cov_A = stats[('A', 2)]['COV']
    mu_B = stats[('B', 2)]['ER']
    cov_B = stats[('B', 2)]['COV']

    # Unconstrained (risky only, first 8 assets) via closed form, t=1
    risky_idx = slice(0, 8)
    rA = mu_A[risky_idx]
    CA = cov_A[risky_idx, risky_idx]
    wA, infoA = closed_form_opt_weights(rA, CA, t=1.0)
    print("\nUnconstrained closed-form weights (Period A, t=1, risky only):")
    print(pretty_weights(wA, ASSETS[risky_idx]))

    rB = mu_B[risky_idx]
    CB = cov_B[risky_idx, risky_idx]
    wB, infoB = closed_form_opt_weights(rB, CB, t=1.0)
    print("\nUnconstrained closed-form weights (Period B, t=1, risky only):")
    print(pretty_weights(wB, ASSETS[risky_idx]))

    # Risk-free + risky (using Cash as R_f)
    rf_A = mu_A[-1]
    cash_w_A, risky_w_A = tangency_with_cash(rA, CA, rf_A, t=1.0)
    print("\nTangency-style allocation with Cash as Rf (Period A, t=1):")
    print(f"Cash [D]: {cash_w_A:.6f}")
    print(pretty_weights(risky_w_A, ASSETS[risky_idx]))

    # Constrained SLSQP optimization (weights >=0, sum=1, minimums, growth band)
    bounds = [(0.0, 1.0)] * len(ASSETS)
    cons = build_constraints(num_assets=len(ASSETS), growth_slice=slice(0, 6), min_constraints=True)

    def solve_and_report(mu, cov, period_label):
        res = solve_constrained_mv(mu, cov, bounds=bounds, constraints=cons, w0=np.ones(len(ASSETS))/len(ASSETS))
        if not res.success:
            print(f"[WARN] Optimization for Period {period_label} did not converge: {res.message}")
        w = res.x / np.sum(res.x)
        exp_ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        print(f"\nConstrained MV weights (Period {period_label}):")
        print(pretty_weights(w, ASSETS))
        print(f"Objective: {res.fun:.6f}   Expected Return: {exp_ret:.6f}   Volatility: {vol:.6f}")
        return w

    w_con_A = solve_and_report(mu_A, cov_A, 'A')
    w_con_B = solve_and_report(mu_B, cov_B, 'B')

    # Out-of-sample realized returns for those constrained weights
    for start, end in OOOS:
        df_C = slice_period(df, start, end)  # use simple returns here
        R_real = realized_cumulative_returns(df_C, ASSETS)

        real_A = float(np.dot(w_con_A, R_real))
        real_B = float(np.dot(w_con_B, R_real))
        exp_A = float(np.dot(w_con_A, mu_A))
        exp_B = float(np.dot(w_con_B, mu_B))

        print(f"\n=== Realized vs Expected over {start[:10]} to {end[:10]} ===")
        print(f"Using Period A constrained weights: Realized={real_A:.4f}  Expected(2y model)={exp_A:.4f}")
        print(f"Using Period B constrained weights: Realized={real_B:.4f}  Expected(2y model)={exp_B:.4f}")

if __name__ == '__main__':
    main()
