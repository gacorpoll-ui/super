import json
import pandas as pd
from collections import Counter

def load_feedback_data(filepath="trade_feedback.json"):
    """Memuat data feedback trading dari file JSON."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error: File '{filepath}' tidak ditemukan atau formatnya tidak valid.")
        return []

def evaluate_weights(trades, weights):
    """
    Mengevaluasi kinerja sebuah set bobot terhadap data trade historis.
    Mengembalikan total profit.
    """
    total_profit = 0
    trades_taken = 0

    for trade in trades:
        components = trade.get('score_components')
        if not components:
            continue

        # Rekalkulasi skor berdasarkan bobot baru
        new_score = sum(count * weights.get(name, 0) for name, count in components.items())

        # Asumsikan sinyal diambil jika skor > 0 untuk BUY dan < 0 untuk SELL
        # Dan asumsikan sinyal asli (yang menghasilkan PnL) diambil
        original_direction = "BUY" if trade['pnl'] > 0 else "SELL" # Ini asumsi kasar

        # Logika sederhana: jika arah sinyal baru sama dengan arah sinyal asli,
        # kita asumsikan trade itu akan diambil dan menghasilkan PnL yang sama.
        if (new_score > 0 and original_direction == "BUY") or (new_score < 0 and original_direction == "SELL"):
            total_profit += trade['pnl']
            trades_taken += 1

    return total_profit, trades_taken

def main():
    """
    Fungsi utama untuk menjalankan optimisasi bobot.
    """
    trades = load_feedback_data()
    if not trades:
        return

    print(f"Ditemukan {len(trades)} record feedback untuk dianalisis.")

    # --- Definisikan beberapa set bobot untuk diuji ---
    # Bobot asli bisa diambil dari config.json sebagai basis
    test_weights = {
        "original": {
            "BULLISH_BOS": 3.0, "BEARISH_BOS": -3.0, "HH": 1.0, "LL": -1.0, "HL": 1.0, "LH": -1.0,
            "FVG_BULLISH": 3.0, "FVG_BEARISH": -3.0, "BULLISH_LS": 3.0, "BEARISH_LS": -3.0,
            "BULLISH_OB": 1.0, "BEARISH_OB": -1.0, "ENGULFING_BULL": 1.0, "ENGULFING_BEAR": -1.0,
            "PINBAR_BULL": 0.8, "PINBAR_BEAR": -0.8, "RBR": 2.0, "DBD": -2.0
        },
        "structure_focused": {
            "BULLISH_BOS": 5.0, "BEARISH_BOS": -5.0, "HH": 2.0, "LL": -2.0, "HL": 2.0, "LH": -2.0,
            "FVG_BULLISH": 1.0, "FVG_BEARISH": -1.0, "BULLISH_LS": 1.0, "BEARISH_LS": -1.0,
            "BULLISH_OB": 0.5, "BEARISH_OB": -0.5, "ENGULFING_BULL": 0.5, "ENGULFING_BEAR": -0.5,
            "PINBAR_BULL": 0.2, "PINBAR_BEAR": -0.2, "RBR": 1.0, "DBD": -1.0
        },
        "zone_focused": {
            "BULLISH_BOS": 1.0, "BEARISH_BOS": -1.0, "HH": 0.5, "LL": -0.5, "HL": 0.5, "LH": -0.5,
            "FVG_BULLISH": 5.0, "FVG_BEARISH": -5.0, "BULLISH_LS": 4.0, "BEARISH_LS": -4.0,
            "BULLISH_OB": 3.0, "BEARISH_OB": -3.0, "ENGULFING_BULL": 1.0, "ENGULFING_BEAR": -1.0,
            "PINBAR_BULL": 0.8, "PINBAR_BEAR": -0.8, "RBR": 2.0, "DBD": -2.0
        }
    }

    print("\n--- Memulai Evaluasi Bobot ---")
    results = []
    for name, weights in test_weights.items():
        profit, num_trades = evaluate_weights(trades, weights)
        results.append({
            "name": name,
            "profit": profit,
            "trades": num_trades
        })
        print(f"Hasil untuk set bobot '{name}': Profit = {profit:.2f} dari {num_trades} trade.")

    # --- Tentukan set bobot terbaik ---
    if results:
        best_result = max(results, key=lambda x: x['profit'])
        print("\n--- Hasil Terbaik ---")
        print(f"Set bobot terbaik adalah: '{best_result['name']}'")
        print(f"Profit: {best_result['profit']:.2f}")
        print(f"Jumlah Trade: {best_result['trades']}")
        print("\nBobot yang direkomendasikan:")
        print(json.dumps(test_weights[best_result['name']], indent=2))
        print("\nAnda bisa menyalin bobot ini ke 'base_weights' di config.json.")

if __name__ == "__main__":
    main()
