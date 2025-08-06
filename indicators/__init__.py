"""
Indikator Package
=================

Paket ini mengumpulkan semua fungsi analisis teknikal, dipecah menjadi
modul-modul yang logis untuk kemudahan pengelolaan.

Struktur:
- zones: Fair Value Gaps (FVG), Order Blocks (OB).
- structure: Market Structure (BOS, HH, LL).
- patterns: Pola Candlestick (Engulfing, Pinbar, dll).
- liquidity: Liquidity Sweeps, Equal Highs/Lows.
- utils: Kalkulator utilitas (ATR, Pivot, OTE, dll) dan Ekstraktor Fitur.
"""

from .zones import (
    detect_order_blocks_multi,
    detect_fvg_multi,
)
from .structure import (
    detect_structure,
)
from .patterns import (
    detect_pinbar,
    detect_engulfing,
    detect_continuation_patterns,
    detect_volume_spike,
)
from .liquidity import (
    detect_eqh_eql,
    detect_liquidity_sweep,
)
from .utils import (
    calculate_atr_dynamic,
    get_daily_high_low,
    get_pivot_points,
    calculate_optimal_trade_entry,
    extract_features_full,
    plot_zones,
    log_zone_events,
    generate_label_fvg,
)

__all__ = [
    'detect_order_blocks_multi',
    'detect_fvg_multi',
    'detect_structure',
    'detect_pinbar',
    'detect_engulfing',
    'detect_continuation_patterns',
    'detect_volume_spike',
    'detect_eqh_eql',
    'detect_liquidity_sweep',
    'calculate_atr_dynamic',
    'get_daily_high_low',
    'get_pivot_points',
    'calculate_optimal_trade_entry',
    'extract_features_full',
    'plot_zones',
    'log_zone_events',
    'generate_label_fvg',
]
