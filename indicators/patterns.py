"""
patterns.py
===========
Deteksi pola-pola candlestick klasik seperti Pinbar, Engulfing,
Rally-Base-Rally (RBR), dan Drop-Base-Drop (DBD).
"""

import pandas as pd
from typing import List, Dict, Any

def detect_pinbar(df: pd.DataFrame, min_ratio: float=2.0) -> List[Dict[str, Any]]:
    """Pinbar: ekor minimal 2x body."""
    pattern = []
    for i in range(2, len(df)):
        c = df.iloc[i]
        body = abs(c['close'] - c['open'])
        upper = c['high'] - max(c['close'], c['open'])
        lower = min(c['close'], c['open']) - c['low']
        # Bullish pinbar
        if lower > min_ratio * body and upper < 0.5 * body:
            pattern.append({'type': 'PINBAR_BULL', 'time': c['time'], 'idx': i})
        # Bearish pinbar
        elif upper > min_ratio * body and lower < 0.5 * body:
            pattern.append({'type': 'PINBAR_BEAR', 'time': c['time'], 'idx': i})
    return pattern

def detect_engulfing(df: pd.DataFrame) -> List[Dict[str, Any]]:
    pattern = []
    for i in range(1, len(df)):
        prev, c = df.iloc[i-1], df.iloc[i]
        # Bullish engulfing
        if c['close'] > c['open'] and prev['close'] < prev['open'] and c['close'] > prev['open'] and c['open'] < prev['close']:
            pattern.append({'type': 'ENGULFING_BULL', 'time': c['time'], 'idx': i})
        # Bearish engulfing
        elif c['close'] < c['open'] and prev['close'] > prev['open'] and c['close'] < prev['open'] and c['open'] > prev['close']:
            pattern.append({'type': 'ENGULFING_BEAR', 'time': c['time'], 'idx': i})
    return pattern

def detect_volume_spike(df: pd.DataFrame, window: int=20, threshold: float=2.0) -> List[Dict[str, Any]]:
    if 'tick_volume' not in df.columns:
        return []
    pattern = []
    vol_ma = df['tick_volume'].rolling(window).mean()
    for i in range(window, len(df)):
        if df['tick_volume'].iloc[i] > threshold * vol_ma.iloc[i]:
            pattern.append({'type': 'VOLUME_SPIKE', 'time': df['time'].iloc[i], 'idx': i})
    return pattern

def detect_continuation_patterns(df: pd.DataFrame, base_max_candles: int = 4, body_threshold: float = 1.2) -> List[Dict[str, Any]]:
    """Mendeteksi pola RBR (Rally-Base-Rally) dan DBD (Drop-Base-Drop)."""
    patterns = []
    if len(df) < base_max_candles + 2:
        return patterns

    df = df.copy()
    df['body'] = abs(df['close'] - df['open'])
    avg_body = df['body'].rolling(20).mean()

    for i in range(len(df) - base_max_candles - 1):
        # Cek RBR (Rally-Base-Rally)
        rally1 = df.iloc[i]
        is_strong_rally1 = rally1['close'] > rally1['open'] and rally1['body'] > avg_body.iloc[i] * body_threshold

        if is_strong_rally1:
            base_candles = df.iloc[i+1 : i+1+base_max_candles]
            base_high = base_candles['high'].max()
            base_low = base_candles['low'].min()

            # Base harus berada dalam rentang rally1 dan tidak terlalu besar
            if base_high < rally1['high'] + (rally1['body'] * 0.2) and base_low > rally1['low'] - (rally1['body'] * 0.2):
                for j in range(1, base_max_candles + 1):
                    rally2_idx = i + j
                    if rally2_idx < len(df) -1:
                        rally2 = df.iloc[rally2_idx + 1]
                        is_strong_rally2 = rally2['close'] > rally2['open'] and rally2['body'] > avg_body.iloc[rally2_idx+1] * body_threshold
                        if is_strong_rally2 and rally2['close'] > rally1['high']:
                            patterns.append({'type': 'RBR', 'time': rally2['time'], 'idx': rally2_idx + 1})
                            i = rally2_idx + 1 # Skip untuk menghindari deteksi tumpang tindih
                            break

        # Cek DBD (Drop-Base-Drop)
        drop1 = df.iloc[i]
        is_strong_drop1 = drop1['close'] < drop1['open'] and drop1['body'] > avg_body.iloc[i] * body_threshold

        if is_strong_drop1:
            base_candles = df.iloc[i+1 : i+1+base_max_candles]
            base_high = base_candles['high'].max()
            base_low = base_candles['low'].min()

            # Base harus berada dalam rentang drop1
            if base_high < drop1['high'] + (drop1['body'] * 0.2) and base_low > drop1['low'] - (drop1['body'] * 0.2):
                for j in range(1, base_max_candles + 1):
                    drop2_idx = i + j
                    if drop2_idx < len(df) - 1:
                        drop2 = df.iloc[drop2_idx + 1]
                        is_strong_drop2 = drop2['close'] < drop2['open'] and drop2['body'] > avg_body.iloc[drop2_idx+1] * body_threshold
                        if is_strong_drop2 and drop2['close'] < drop1['low']:
                            patterns.append({'type': 'DBD', 'time': drop2['time'], 'idx': drop2_idx + 1})
                            i = drop2_idx + 1 # Skip
                            break

    return patterns
