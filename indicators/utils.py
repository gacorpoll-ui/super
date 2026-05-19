"""
utils.py
========
Kumpulan fungsi utilitas untuk analisis teknikal, termasuk:
- Kalkulator: ATR, Pivot Points, Optimal Trade Entry (OTE).
- Ekstraktor Fitur: Fungsi untuk mengubah data mentah menjadi vektor fitur untuk AI.
- Alat Bantu: Plotting, logging, dan pembuatan label otomatis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def calculate_atr_dynamic(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return 0.0
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    atr_val = tr.ewm(span=period, adjust=False).mean().iloc[-1]
    return float(atr_val)

def get_daily_high_low(df: pd.DataFrame) -> Dict[str, Any]:
    today = pd.to_datetime(df['time'].iloc[-1]).date()
    today_data = df[pd.to_datetime(df['time']).dt.date == today]
    high = today_data['high'].max()
    low = today_data['low'].min()
    last_close = today_data['close'].iloc[-1] if len(today_data) else df['close'].iloc[-1]
    return {
        'daily_high': high,
        'daily_low': low,
        'distance_to_high': abs(last_close - high),
        'distance_to_low': abs(last_close - low)
    }

def get_pivot_points(df: pd.DataFrame) -> Dict[str, float]:
    # Classic pivot, pakai data daily terakhir
    today = pd.to_datetime(df['time'].iloc[-1]).date()
    prev_day = today - pd.Timedelta(days=1)
    prev_data = df[pd.to_datetime(df['time']).dt.date == prev_day]
    if len(prev_data) == 0:
        prev_data = df.iloc[-20:]  # fallback: 20 bar terakhir
    high, low, close = prev_data['high'].max(), prev_data['low'].min(), prev_data['close'].iloc[-1]
    pivot = (high + low + close) / 3
    r1 = 2*pivot - low
    s1 = 2*pivot - high
    return {'pivot': pivot, 'r1': r1, 's1': s1}

def calculate_optimal_trade_entry(swing_start_price: float, swing_end_price: float, direction: str) -> Dict[str, float]:
    fib_levels = {
        '0.618': 0.618,
        '0.705': 0.705,
        '0.790': 0.790
    }
    ote_zone: Dict[str, float] = {}
    high_val = max(swing_start_price, swing_end_price)
    low_val = min(swing_start_price, swing_end_price)
    price_range = high_val - low_val
    if direction.upper() == "BUY":
        ote_zone['upper'] = high_val - (price_range * fib_levels['0.618'])
        ote_zone['mid'] = high_val - (price_range * fib_levels['0.705'])
        ote_zone['lower'] = high_val - (price_range * fib_levels['0.790'])
    elif direction.upper() == "SELL":
        ote_zone['upper'] = low_val + (price_range * fib_levels['0.790'])
        ote_zone['mid'] = low_val + (price_range * fib_levels['0.705'])
        ote_zone['lower'] = low_val + (price_range * fib_levels['0.618'])
    return ote_zone

def extract_features_full(
    df: pd.DataFrame,
    structure: str,
    order_blocks: List[Dict[str, Any]],
    fvg_zones: List[Dict[str, Any]],
    patterns: List[Dict[str, Any]],
    boundary: Dict[str, Any],
    pivot: Dict[str, float]
) -> np.ndarray:
    """Keluarkan vektor fitur komprehensif untuk AI/ML."""
    last_price = df['close'].iloc[-1]
    features = [
        last_price,
        calculate_atr_dynamic(df),
        min([ob['distance'] for ob in order_blocks]) if order_blocks else 0,
        max([ob['strength'] for ob in order_blocks]) if order_blocks else 0,
        min([fvg['distance'] for fvg in fvg_zones]) if fvg_zones else 0,
        max([fvg['strength'] for fvg in fvg_zones]) if fvg_zones else 0,
        int(any(p['type'].startswith('ENGULFING') for p in patterns)),
        boundary.get('distance_to_high', 0),
        boundary.get('distance_to_low', 0),
        pivot.get('r1', 0) - last_price,
        pivot.get('s1', 0) - last_price,
    ]
    return np.array(features, dtype=float)

def generate_label_fvg(df: pd.DataFrame, fvg_zones: List[Dict[str, Any]], horizon: int=10, pip_thresh: float=0.001) -> List[int]:
    """
    Label 1 jika X bar setelah FVG harga naik/turun >= pip_thresh, else 0.
    """
    label = [0]*len(df)
    for fvg in fvg_zones:
        idx = df[df['time']==fvg['time']].index[0]
        base_price = (fvg['start'] + fvg['end'])/2
        future_idx = min(idx + horizon, len(df)-1)
        if 'BULLISH' in fvg['type']:
            if df['high'].iloc[future_idx] - base_price >= pip_thresh:
                label[idx] = 1
        elif 'BEARISH' in fvg['type']:
            if base_price - df['low'].iloc[future_idx] >= pip_thresh:
                label[idx] = 1
    return label

def plot_zones(df, order_blocks, fvg_zones, patterns=None):
    plt.figure(figsize=(15, 7))
    plt.plot(df['time'], df['close'], label='Close Price', color='black')
    for ob in order_blocks:
        plt.axhspan(ob['low'], ob['high'], color='green' if 'BULL' in ob['type'] else 'red', alpha=0.25)
    for fvg in fvg_zones:
        plt.axhspan(fvg['start'], fvg['end'], color='blue' if 'BULL' in fvg['type'] else 'orange', alpha=0.15)
    if patterns:
        for p in patterns:
            idx = p['idx']
            plt.scatter(df['time'].iloc[idx], df['close'].iloc[idx], marker='^' if 'BULL' in p['type'] else 'v', color='purple')
    plt.legend()
    plt.title("Price with OB/FVG/Pattern Zones")
    plt.show()

def log_zone_events(order_blocks, fvg_zones, patterns):
    """Log bar yang trigger event, useful untuk AI/analitik."""
    event_log = []
    for ob in order_blocks:
        event_log.append({'time': ob['time'], 'price': (ob['high']+ob['low'])/2, 'type': ob['type']})
    for fvg in fvg_zones:
        event_log.append({'time': fvg['time'], 'price': (fvg['start']+fvg['end'])/2, 'type': fvg['type']})
    for p in patterns:
        event_log.append({'time': p['time'], 'price': None, 'type': p['type']})
    event_log = sorted(event_log, key=lambda x: x['time'])
    return event_log
