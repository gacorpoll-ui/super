"""
structure.py
============
Deteksi struktur pasar seperti Higher Highs (HH), Lower Lows (LL),
dan Break of Structure (BOS).
"""

import pandas as pd
from typing import Dict, Any, Tuple

def detect_structure(df: pd.DataFrame, swing_lookback: int = 5) -> Tuple[str, Dict[str, Any]]:
    if len(df) < swing_lookback * 2:
        return "INDECISIVE", {}

    df = df.copy()

    # Deteksi Swing Highs and Lows yang lebih robust
    df['is_swing_high'] = (df['high'] == df['high'].rolling(swing_lookback*2+1, center=True).max())
    df['is_swing_low'] = (df['low'] == df['low'].rolling(swing_lookback*2+1, center=True).min())

    swing_highs = df[df['is_swing_high']]
    swing_lows = df[df['is_swing_low']]

    res = []
    last_swing_high_price = None
    last_swing_low_price = None
    last_swing_high_idx = None
    last_swing_low_idx = None

    # Urutkan swing points berdasarkan waktu (index)
    if not swing_highs.empty:
        last_sh = swing_highs.iloc[-1]
        last_swing_high_price = last_sh['high']
        last_swing_high_idx = last_sh.name
        if len(swing_highs) > 1:
            prev_sh = swing_highs.iloc[-2]
            if last_sh['high'] > prev_sh['high']: res.append('HH')
            else: res.append('LH')

    if not swing_lows.empty:
        last_sl = swing_lows.iloc[-1]
        last_swing_low_price = last_sl['low']
        last_swing_low_idx = last_sl.name
        if len(swing_lows) > 1:
            prev_sl = swing_lows.iloc[-2]
            if last_sl['low'] > prev_sl['low']: res.append('HL')
            else: res.append('LL')

    # --- LOGIKA BARU UNTUK BREAK OF STRUCTURE (BOS) ---
    # Membutuhkan penutupan candle (candle close) untuk konfirmasi
    if last_swing_high_idx is not None:
        # Cek candle SETELAH swing high terakhir
        subsequent_candles = df.loc[last_swing_high_idx:].iloc[1:]
        confirmed_break = subsequent_candles[subsequent_candles['close'] > last_swing_high_price]
        if not confirmed_break.empty:
            res.append("BULLISH_BOS")

    if last_swing_low_idx is not None:
        # Cek candle SETELAH swing low terakhir
        subsequent_candles = df.loc[last_swing_low_idx:].iloc[1:]
        confirmed_break = subsequent_candles[subsequent_candles['close'] < last_swing_low_price]
        if not confirmed_break.empty:
            res.append("BEARISH_BOS")

    structure_str = "/".join(sorted(list(set(res)))) if res else "INDECISIVE"
    return structure_str, {'last_high': last_swing_high_price, 'last_low': last_swing_low_price}
