"""
liquidity.py
============
Deteksi konsep-konsep terkait likuiditas pasar, seperti
Equal Highs/Lows (EQH/EQL) dan Liquidity Sweeps (LS).
"""

import pandas as pd
from typing import List, Dict, Any, Tuple

def detect_eqh_eql(df: pd.DataFrame, window: int = 10, tolerance: float = 0.0003) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    if len(df) < window:
        return [], []
    eqh, eql = [], []
    for i in range(window, len(df)):
        hi = df['high'].iloc[i-window:i]
        lo = df['low'].iloc[i-window:i]
        if (hi.max() - hi.min()) < (hi.mean() * tolerance):
            eqh.append({'time': df['time'].iloc[i], 'value': float(hi.mean())})
        if (lo.max() - lo.min()) < (lo.mean() * tolerance):
            eql.append({'time': df['time'].iloc[i], 'value': float(lo.mean())})
    return eqh, eql

def detect_liquidity_sweep(df: pd.DataFrame, min_candles_for_swing: int = 5) -> List[Dict[str, Any]]:
    liquidity_sweeps = []
    if len(df) < min_candles_for_swing * 2:
        return liquidity_sweeps
    df = df.copy()
    df['is_swing_high'] = (df['high'].rolling(min_candles_for_swing, center=True).max() == df['high'])
    df['is_swing_low'] = (df['low'].rolling(min_candles_for_swing, center=True).min() == df['low'])
    swing_highs_idx = df[df['is_swing_high']].index.tolist()
    swing_lows_idx = df[df['is_swing_low']].index.tolist()
    current_candle_idx = len(df) - 1
    for low_idx in reversed(swing_lows_idx):
        if low_idx < current_candle_idx - 1:
            prev_swing_low_val = df['low'].loc[low_idx]
            for i in range(low_idx + 1, current_candle_idx + 1):
                if df['low'].iloc[i] < prev_swing_low_val and df['close'].iloc[i] > prev_swing_low_val:
                    if df['close'].iloc[current_candle_idx] > df['open'].iloc[current_candle_idx]:
                        liquidity_sweeps.append({
                            'type': 'BULLISH_LS',
                            'swept_level': float(prev_swing_low_val),
                            'sweep_time': df['time'].loc[i],
                            'current_time': df['time'].iloc[current_candle_idx]
                        })
                        break
    for high_idx in reversed(swing_highs_idx):
        if high_idx < current_candle_idx - 1:
            prev_swing_high_val = df['high'].loc[high_idx]
            for i in range(high_idx + 1, current_candle_idx + 1):
                if df['high'].iloc[i] > prev_swing_high_val and df['close'].iloc[i] < prev_swing_high_val:
                    if df['close'].iloc[current_candle_idx] < df['open'].iloc[current_candle_idx]:
                        liquidity_sweeps.append({
                            'type': 'BEARISH_LS',
                            'swept_level': float(prev_swing_high_val),
                            'sweep_time': df['time'].loc[i],
                            'current_time': df['time'].iloc[current_candle_idx]
                        })
                        break
    return liquidity_sweeps
