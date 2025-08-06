"""
zones.py
========
Deteksi zona-zona penting di pasar seperti Order Blocks (OB) dan Fair Value Gaps (FVG).
"""

import pandas as pd
from typing import List, Dict, Any, Optional

# Impor utilitas dari modul lain dalam package yang sama
from .utils import calculate_atr_dynamic

def detect_order_blocks_multi(
    df: pd.DataFrame, lookback: int = 15, structure_filter: Optional[str]=None, max_age: int=20
) -> List[Dict[str, Any]]:
    order_blocks = []
    now_price = df['close'].iloc[-1]
    atr = calculate_atr_dynamic(df)
    last_idx = df.index[-1]
    for i in range(len(df)-lookback-2, len(df)-2):
        candle = df.iloc[i]
        age = last_idx - df.index[i]
        if age > max_age:
            continue
        # Bullish OB
        if candle['close'] < candle['open']:
            found_bos = False
            for j in range(i+1, min(i+lookback+1, len(df))):
                if df.iloc[j]['close'] > candle['high']:
                    found_bos = True
                    break
            if found_bos:
                if structure_filter and structure_filter != "BULLISH_BOS":
                    continue
                strength = (abs(candle['open'] - candle['close']) / atr) if atr else 1
                order_blocks.append({
                    'type': 'BULLISH_OB',
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'time': candle['time'],
                    'age': age,
                    'strength': strength,
                    'distance': abs(now_price - ((candle['open']+candle['close'])/2)),
                })
        # Bearish OB
        elif candle['close'] > candle['open']:
            found_bos = False
            for j in range(i+1, min(i+lookback+1, len(df))):
                if df.iloc[j]['close'] < candle['low']:
                    found_bos = True
                    break
            if found_bos:
                if structure_filter and structure_filter != "BEARISH_BOS":
                    continue
                strength = (abs(candle['open'] - candle['close']) / atr) if atr else 1
                order_blocks.append({
                    'type': 'BEARISH_OB',
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'time': candle['time'],
                    'age': age,
                    'strength': strength,
                    'distance': abs(now_price - ((candle['open']+candle['close'])/2)),
                })
    order_blocks = sorted(order_blocks, key=lambda x: (x['distance'], -x['strength'], x['age']))
    return order_blocks

def detect_fvg_multi(df: pd.DataFrame, min_gap: float=0.0002, max_age: int=20) -> List[Dict[str, Any]]:
    fvg_zones = []
    now_price = df['close'].iloc[-1]
    atr = calculate_atr_dynamic(df)
    last_idx = df.index[-1]
    for i in range(1, len(df)-1):
        c_prev, c_now, c_next = df.iloc[i-1], df.iloc[i], df.iloc[i+1]
        age = last_idx - df.index[i]
        if age > max_age:
            continue
        # Bullish FVG
        gap = c_next['low'] - c_prev['high']
        if gap > min_gap:
            strength = gap / atr if atr else 1
            fvg_zones.append({
                'type': 'FVG_BULLISH',
                'start': float(c_prev['high']),
                'end': float(c_next['low']),
                'time': c_now['time'],
                'age': age,
                'strength': strength,
                'distance': abs(now_price - ((c_prev['high']+c_next['low'])/2))
            })
        # Bearish FVG
        gap = c_prev['low'] - c_next['high']
        if gap > min_gap:
            strength = gap / atr if atr else 1
            fvg_zones.append({
                'type': 'FVG_BEARISH',
                'start': float(c_next['high']),
                'end': float(c_prev['low']),
                'time': c_now['time'],
                'age': age,
                'strength': strength,
                'distance': abs(now_price - ((c_next['high']+c_prev['low'])/2))
            })
    fvg_zones = sorted(fvg_zones, key=lambda x: (x['distance'], -x['strength'], x['age']))
    return fvg_zones
