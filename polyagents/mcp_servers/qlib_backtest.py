"""Backtest MCP — factor → model → backtest over the polyagents SQLite history.

The Merakku doc's P0 backtest core (Qlib). qlib lives in its OWN venv, so this
server is **standalone** (no polyagents imports): it reads the SQLite DB by path
and uses the qlib venv's pandas + lightgbm/scikit-learn. Run it with the qlib
interpreter:

    C:\\qlib\\.venv\\Scripts\\python.exe -m polyagents.mcp_servers.qlib_backtest
    (or .../python.exe polyagents/mcp_servers/qlib_backtest.py)

Discipline: the train/test split is **time-based** (train on earlier timestamps,
test on later) so there's no look-ahead leakage — the one rule you cannot break
when backtesting. Full qlib Alpha158 handlers are a future upgrade; this is a
real, leakage-safe factor→model→backtest to validate whether a signal has edge.
"""
from __future__ import annotations

import os
import sqlite3
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("qlib-backtest")

_DB = os.getenv("POLYAGENTS_DB_PATH",
                os.path.join(os.path.expanduser("~"), ".polyagents", "cache", "polyagents.db"))


def _load_candles():
    import pandas as pd

    if not os.path.exists(_DB):
        return pd.DataFrame()
    con = sqlite3.connect(_DB)
    try:
        df = pd.read_sql_query(
            "SELECT token_id, ts, close, volume FROM candles ORDER BY token_id, ts", con)
    finally:
        con.close()
    return df


def _features(df, fwd: int):
    """Per-bar factors + a forward-return label, grouped by token (no leakage)."""
    import numpy as np
    import pandas as pd

    frames = []
    for tok, g in df.groupby("token_id"):
        g = g.sort_values("ts").reset_index(drop=True)
        if len(g) < fwd + 12:
            continue
        c = g["close"]
        feat = pd.DataFrame({
            "ts": g["ts"],
            "ret1": c.pct_change(1),
            "ret5": c.pct_change(5),
            "mom10": c / c.rolling(10).mean() - 1.0,
            "vol5": c.pct_change().rolling(5).std(),
            "volume": g["volume"],
        })
        fwd_ret = c.shift(-fwd) / c - 1.0
        feat["fwd_ret"] = fwd_ret
        feat["y"] = (fwd_ret > 0).astype(int)
        frames.append(feat.dropna())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()


@mcp.tool()
def data_summary() -> dict:
    """How much history is in the SQLite DB available to backtest."""
    df = _load_candles()
    if df.empty:
        return {"db": _DB, "exists": os.path.exists(_DB), "tokens": 0, "candles": 0}
    return {"db": _DB, "exists": True, "tokens": int(df["token_id"].nunique()),
            "candles": int(len(df))}


@mcp.tool()
def run_backtest(forward_bars: int = 5, train_frac: float = 0.7) -> dict:
    """Train a model on early history and test on later history (leakage-safe),
    predicting the sign of the ``forward_bars``-ahead return. Reports accuracy,
    information coefficient (IC), and a simple long/short backtest Sharpe."""
    import numpy as np

    data = _features(_load_candles(), forward_bars)
    if len(data) < 100:
        return {"error": "not enough history yet — collect more markets first",
                "samples": int(len(data))}
    data = data.sort_values("ts").reset_index(drop=True)
    cut = int(len(data) * train_frac)
    cols = ["ret1", "ret5", "mom10", "vol5", "volume"]
    Xtr, ytr = data.loc[:cut, cols], data.loc[:cut, "y"]
    Xte = data.loc[cut:, cols]
    yte = data.loc[cut:, "y"].to_numpy()
    fwd = data.loc[cut:, "fwd_ret"].to_numpy()

    try:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(n_estimators=120, num_leaves=15, verbose=-1)
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        engine = "lightgbm"
    except Exception:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=500).fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        engine = "logistic"

    pred = (proba > 0.5).astype(int)
    acc = float((pred == yte).mean())
    ic = float(np.corrcoef(proba, fwd)[0, 1]) if len(set(proba)) > 1 else 0.0
    # simple strategy: long if proba>0.55, short if <0.45; per-trade signed fwd return
    pos = np.where(proba > 0.55, 1.0, np.where(proba < 0.45, -1.0, 0.0))
    rets = pos * fwd
    traded = rets[pos != 0]
    sharpe = float(traded.mean() / traded.std() * np.sqrt(252)) if traded.size > 1 and traded.std() else 0.0
    # equity curve over executed trades (for plotting), downsampled to ~80 points
    eq = np.cumprod(1.0 + traded) if traded.size else np.array([1.0])
    step = max(1, len(eq) // 80)
    curve = [round(float(x), 4) for x in eq[::step]]
    if curve and curve[-1] != round(float(eq[-1]), 4):
        curve.append(round(float(eq[-1]), 4))
    return {
        "engine": engine, "forward_bars": forward_bars,
        "train_samples": int(cut), "test_samples": int(len(yte)),
        "accuracy": round(acc, 4), "information_coefficient": round(ic, 4),
        "n_trades": int(traded.size), "avg_trade_return": round(float(traded.mean()) if traded.size else 0.0, 5),
        "total_return": round(float(eq[-1] - 1.0), 4) if traded.size else 0.0,
        "sharpe_annualized": round(sharpe, 3),
        "equity_curve": curve,
        "verdict": "edge" if ic > 0.03 and acc > 0.52 else "no clear edge (likely noise)",
    }


def main() -> None:
    mcp.run(transport="streamable-http" if "--http" in sys.argv else "stdio")


if __name__ == "__main__":
    main()
