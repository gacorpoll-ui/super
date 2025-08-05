# server_comm.py
#
# Deskripsi:
# Versi ini telah disempurnakan untuk mendukung semua tipe order:
# Market (Buy/Sell Entry), Stop (Buy/Sell Stop), dan Limit (Buy/Sell Limit).

from __future__ import annotations
import logging
import requests
from typing import Dict

def send_signal_to_server(**payload: Any) -> str:
    """Mengirim sinyal trading ke server dan mengembalikan status keberhasilan."""
    
    # Ekstrak URL dan hapus dari payload agar tidak terkirim sebagai JSON
    server_url = payload.pop("server_url", None)
    if not server_url:
        logging.error("server_url tidak ditemukan di payload. Sinyal tidak dikirim.")
        return 'FAILED'
        
    signal_json = payload.get("signal_json", {})
    if not isinstance(signal_json, dict):
        logging.error("Tipe data signal_json tidak valid (harus dictionary). Sinyal tidak dikirim.")
        return 'FAILED'
    
    order_type = payload.get("order_type", "WAIT")
    symbol = payload.get("symbol", "UNKNOWN")

    # Menentukan `signal_type` umum berdasarkan `order_type`
    if "CANCEL" in order_type.upper() or "DELETE" in order_type.upper():
        payload['signal'] = "CANCEL"
    elif "BUY" in order_type.upper():
        payload['signal'] = "BUY"
    elif "SELL" in order_type.upper():
        payload['signal'] = "SELL"
    else:
        payload['signal'] = "WAIT"

    try:
        response = requests.post(server_url, json=payload, timeout=10)
        log_message = f"Sinyal {payload.get('signal', 'UNKNOWN')} untuk {symbol} dikirim."

        if response.status_code == 200:
            logging.info(f"✅ {log_message} Status: BERHASIL.")
            return 'SUCCESS'
        elif 400 <= response.status_code < 500:
            logging.error(f"❌ {log_message} Status: DITOLAK. Respons: {response.text}")
            return 'REJECTED'
        else:
            logging.error(f"❌ {log_message} Status: GAGAL. Kode: {response.status_code}, Respons: {response.text}")
            return 'FAILED'
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Error koneksi saat mengirim sinyal: {e}")
        return 'FAILED'

def cancel_signal(signal_id: str, active_signals: Dict[str, Dict[str, any]], api_key: str, server_url: str, secret_key: str) -> None:
    """Membangun dan mengirim sinyal pembatalan untuk semua tipe order (Market, Limit, Stop)."""
    if signal_id not in active_signals:
        return

    original = active_signals[signal_id]['signal_json']
    symbol = original.get("Symbol")

    entry_val = (original.get("BuyEntry") or original.get("SellEntry") or
                 original.get("BuyStop") or original.get("SellStop") or
                 original.get("BuyLimit") or original.get("SellLimit"))

    if not symbol or not entry_val:
        logging.error(f"Data tidak lengkap untuk membatalkan sinyal ID {signal_id}.")
        return

    cancel_json = {
        "Symbol": symbol,
        "DeleteLimit/Stop": entry_val,
        "BuyEntry": "", "BuySL": "", "BuyTP": "", "SellEntry": "", "SellSL": "", "SellTP": "",
        "BuyStop": "", "BuyStopSL": "", "BuyStopTP": "", "SellStop": "", "SellStopSL": "", "SellStopTP": "",
        "BuyLimit": "", "BuyLimitSL": "", "BuyLimitTP": "", "SellLimit": "", "SellLimitSL": "", "SellLimitTP": "",
    }

    send_signal_to_server(symbol, cancel_json, api_key, server_url, secret_key)
    del active_signals[signal_id]