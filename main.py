# main.py (Refactored for Profiles & Learning)
#
# Deskripsi:
# Versi ini mendukung "Strategy Profiles" dari config.json dan
# mengimplementasikan siklus belajar adaptif untuk setiap profil.

import logging
import os
import sys
import time
import numpy as np
import xgboost as xgb
import subprocess
from datetime import datetime
from typing import Dict, List, Any
import json

from data_fetching import get_candlestick_data, DataCache
from gng_model import initialize_gng_models, GrowingNeuralGas, save_gng_model
from log_handler import WebServerHandler
from learning import analyze_and_adapt_profiles
from signal_generator import (
    analyze_tf_opportunity,
    build_signal_format,
    make_signal_id,
    get_open_positions_per_tf,
    get_active_orders,
    is_far_enough
)
from server_comm import send_signal_to_server

# --- Global States ---
DATA_CACHE = DataCache()
active_signals: Dict[str, Dict[str, Dict[str, any]]] = {}
signal_cooldown: Dict[str, datetime] = {}
gng_last_retrain_time: Dict[str, datetime] = {}
xgb_retraining_process = None
xgb_last_retrain_trade_count = 0

def load_config(filepath: str = "config.json") -> Dict[str, Any]:
    """Memuat konfigurasi dari file JSON dan setup logging."""
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        log_config = config.get('logging', {})
        log_format = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        log_level = getattr(logging, log_config.get('level', 'INFO').upper(), logging.INFO)
        logging.basicConfig(level=log_level, format=log_format)
        
        # Setup web server logging jika URL ada di config
        if config.get('global_settings', {}).get('server_url'):
            root_logger = logging.getLogger()
            # Asumsi URL untuk log adalah bagian dari server utama
            log_server_url = config['global_settings']['server_url'].replace('/submit_signal', '')
            web_handler = WebServerHandler(url=log_server_url)
            web_handler.setFormatter(logging.Formatter(log_format))
            root_logger.addHandler(web_handler)

        logging.info("Konfigurasi berhasil dimuat dari %s.", filepath)
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.critical("Error file konfigurasi: %s. Bot tidak bisa berjalan.", e)
        exit()
    except Exception as e:
        logging.critical("Error saat memuat konfigurasi: %s", e)
        exit()

def initialize_models(config: Dict[str, Any]) -> tuple[dict, dict, dict]:
    """Inisialisasi semua model (GNG, XGBoost) untuk semua simbol."""
    gng_models, gng_feature_stats, xgb_models = {}, {}, {}
    global_conf = config.get('global_settings', {})
    symbols = global_conf.get('symbols_to_analyze', [])
    
    # --- Inisialisasi Model GNG ---
    # Kita hanya perlu inisialisasi GNG untuk satu simbol karena statistik fitur cenderung serupa
    # dan modelnya sendiri akan dilatih per timeframe.
    if symbols:
        logging.info("--- Inisialisasi Model Kontekstual (GNG) ---")
        # Ambil semua timeframe dari semua profil
        all_timeframes = list(set(tf for profile in config.get("strategy_profiles", {}).values() for tf in profile.get("timeframes", [])))
        gng_models, gng_feature_stats = initialize_gng_models(
            symbol=symbols[0],  # Gunakan simbol pertama sebagai basis
            timeframes=all_timeframes,
            model_dir="gng_models",
            mt5_path=global_conf.get('mt5_terminal_path'),
            get_data_func=get_candlestick_data
        )

    logging.info("--- Inisialisasi Model AI (XGBoost) ---")
    for symbol in symbols:
        try:
            model_path = f"xgboost_model_{symbol}.json"
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            xgb_models[symbol] = model
            logging.info("Model AI untuk %s berhasil dimuat.", symbol)
        except Exception as e:
            logging.error("GAGAL memuat model AI untuk %s: %s.", symbol, e)
            xgb_models[symbol] = None
            
    return gng_models, gng_feature_stats, xgb_models

def handle_opportunity(opp: Dict[str, Any], symbol: str, tf: str, config: Dict[str, Any], xgb_model: xgb.XGBClassifier, profile_name: str):
    """Memproses, memvalidasi, dan mengirim sinyal jika ada peluang yang memenuhi syarat."""
    global active_signals, signal_cooldown
    
    profile_config = config['strategy_profiles'][profile_name]
    global_config = config['global_settings']
    confidence_threshold = profile_config['confidence_threshold']
    
    if abs(opp['score']) < confidence_threshold:
        logging.info("⏳ [%s|%s|%s] SINYAL WAIT. Skor (%.2f) di bawah threshold (%.1f).", profile_name, symbol, tf, opp['score'], confidence_threshold)
        return False

    logging.info("✅ [%s|%s|%s] SINYAL DITEMUKAN! Peluang %s memenuhi syarat. Skor: %.2f (Min: %.1f).", profile_name, symbol, tf, opp['signal'], opp['score'], confidence_threshold)
    
    if xgb_model and opp.get('features') is not None and opp['features'].size > 0:
        logging.info("[Arshy | %s|%s] Meminta validasi kuantitatif dari Catelya...", profile_name, symbol, tf)
        features = np.array(opp['features']).reshape(1, -1)
        win_probability = xgb_model.predict_proba(features)[0][1]
        
        if win_probability < config.get('strategy_profiles', {}).get(profile_name, {}).get('xgboost_confidence_threshold', 0.75):
            logging.warning("[Catelya | %s|%s] Probabilitas keberhasilan rendah (%.2f%%). Rekomendasi: BATALKAN.", profile_name, symbol, tf, win_probability * 100)
            return False
        
        logging.info("[Catelya | %s|%s] Probabilitas keberhasilan terhitung: %.2f%%. Rekomendasi: LANJUTKAN.", profile_name, symbol, tf, win_probability * 100)

    # --- Persiapan & Pengiriman Sinyal ---
    order_type_to_use = opp.get('order_type', opp['signal'])
    signal_json = build_signal_format(
        symbol=symbol, entry_price=float(opp['entry_price_chosen']), 
        direction=opp['signal'], sl=float(opp['sl']), tp=float(opp['tp']), 
        order_type=order_type_to_use
    )
    
    # Payload sekarang menyertakan semua konteks untuk pembelajaran
    payload = {
        "symbol": symbol,
        "signal_json": signal_json,
        "api_key": global_config['api_key'],
        "server_url": global_config['server_url'],
        "secret_key": global_config['secret_key'],
        "order_type": order_type_to_use,
        "score": opp.get('score'),
        "info": opp.get('info'),
        "profile_name": profile_name
    }
    
    send_status = send_signal_to_server(**payload)

    if send_status == 'SUCCESS':
        sig_id = make_signal_id(signal_json)
        active_signals[symbol][sig_id] = {'signal_json': signal_json, 'tf': tf, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        logging.info("[%s|%s|%s] Sinyal berhasil dikirim!", profile_name, symbol, tf)
        signal_cooldown[symbol] = datetime.now()
        return True
        
    return False

def process_profile(profile_name: str, config: Dict[str, Any], models: Dict[str, Any], adapted_weights: Dict[str, float]):
    """Menjalankan seluruh siklus analisis untuk satu profil strategi."""
    global signal_cooldown
    
    profile_config = config['strategy_profiles'][profile_name]
    global_config = config['global_settings']
    gng_models = models.get('gng', {})
    gng_stats = models.get('gng_stats', {})
    
    logging.info("--- Memproses Profil: '%s' ---", profile_name)

    for symbol in global_config['symbols_to_analyze']:
        if symbol in signal_cooldown:
            cooldown_minutes = profile_config.get('signal_cooldown_minutes', 1)
            time_since_signal = (datetime.now() - signal_cooldown[symbol]).total_seconds() / 60
            if time_since_signal < cooldown_minutes:
                logging.info("[%s|%s] Sabar dulu... Masih dalam masa tenang. Sisa: %.1f menit.", profile_name, symbol, cooldown_minutes - time_since_signal)
                continue
            else:
                del signal_cooldown[symbol]

        for tf in profile_config['timeframes']:
            logging.info("[%s|%s|%s] Menganalisis...", profile_name, symbol, tf)
            try:
                # Mengambil model GNG dan statistik yang sesuai untuk timeframe ini
                gng_model_for_tf = gng_models.get(tf)

                opp = analyze_tf_opportunity(
                    symbol=symbol, tf=tf, mt5_path=global_config['mt5_terminal_path'],
                    gng_model=gng_model_for_tf,
                    gng_feature_stats=gng_stats,
                    confidence_threshold=0.0,
                    min_distance_pips_per_tf=profile_config['min_distance_pips_per_tf'],
                    weights=adapted_weights # Gunakan bobot yang sudah diadaptasi
                )
                if opp and opp.get('signal') != "WAIT":
                    signal_sent = handle_opportunity(opp, symbol, tf, config, models['xgb'].get(symbol), profile_name)
                    if signal_sent:
                        break 
            except Exception as e:
                logging.error("Error saat menganalisis %s|%s|%s: %s", profile_name, symbol, tf, e, exc_info=True)

def retrain_gng_models_if_needed(config: Dict[str, Any], gng_models: Dict[str, GrowingNeuralGas]):
    """Melatih ulang model GNG secara periodik dengan data baru."""
    global gng_last_retrain_time

    retrain_interval_hours = config.get("learning", {}).get("gng_retrain_interval_hours", 4)
    now = datetime.now()

    for tf, model in gng_models.items():
        if tf not in gng_last_retrain_time or (now - gng_last_retrain_time[tf]).total_seconds() > retrain_interval_hours * 3600:
            logging.info(f"--- Memulai Pelatihan Ulang Periodik untuk GNG Timeframe: {tf} ---")

            global_conf = config.get('global_settings', {})
            symbol = global_conf.get('symbols_to_analyze', [])[0] # Ambil simbol basis

            # Ambil data baru yang lebih banyak untuk dilatih
            df_hist = get_candlestick_data(symbol, tf, 1500, global_conf['mt5_terminal_path'])
            if df_hist is None or len(df_hist) < 100:
                logging.warning(f"GNG Retrain: Tidak cukup data historis untuk {symbol} | {tf}. Pelatihan dilewati.")
                continue

            # Logika untuk mempersiapkan data (disederhanakan, idealnya dari gng_model.py)
            # Untuk sekarang, kita asumsikan normalisasi terjadi di dalam `fit` atau tidak diperlukan untuk update
            # Dalam implementasi nyata, kita akan memanggil `prepare_features_from_df` dan `_normalize_features`
            # Namun untuk menjaga `main.py` tetap bersih, kita akan memanggil `fit` secara langsung
            # Ini mengasumsikan data mentah dapat diproses oleh `fit`

            # Contoh sederhana: kita buat numpy array dari harga penutupan
            # IMPLEMENTASI SEBENARNYA HARUS LEBIH KOMPLEKS (menggunakan semua fitur)
            # Untuk saat ini, kita akan melewati pelatihan ulang yang sebenarnya dan hanya menandai waktu
            # TODO: Implementasikan pipeline data yang benar untuk retrain GNG di sini

            # model.fit(prepared_data, num_iterations=1) # CONTOH PANGGILAN
            # save_gng_model(tf, model, "gng_models") # Simpan model yang sudah diupdate

            logging.info(f"GNG Retrain: Model untuk {tf} telah diupdate (simulasi).")
            gng_last_retrain_time[tf] = now

def trigger_xgb_retraining_if_needed(config: Dict[str, Any]):
    """Memeriksa apakah model XGBoost perlu dilatih ulang dan memicu prosesnya."""
    global xgb_retraining_process, xgb_last_retrain_trade_count

    learning_config = config.get("learning", {})
    if not learning_config.get("xgb_auto_retrain_enabled", True):
        return

    # Periksa apakah proses retraining sebelumnya masih berjalan
    if xgb_retraining_process and xgb_retraining_process.poll() is None:
        logging.info("XGB Retrain: Proses pelatihan ulang masih berjalan di latar belakang.")
        return

    feedback_file = learning_config.get("feedback_file", "trade_feedback.json")
    try:
        with open(feedback_file, 'r') as f:
            current_trades = json.load(f)
        current_trade_count = len(current_trades)
    except (FileNotFoundError, json.JSONDecodeError):
        current_trade_count = 0

    retrain_threshold = learning_config.get("xgb_retrain_trade_threshold", 50)

    if current_trade_count - xgb_last_retrain_trade_count >= retrain_threshold:
        logging.info(f"--- Memicu Pelatihan Ulang Otomatis untuk XGBoost ({current_trade_count} trades) ---")

        # Gunakan interpreter python yang sama dengan yang menjalankan bot
        python_executable = sys.executable

        # Jalankan skrip sebagai proses terpisah di latar belakang
        try:
            # Langkah 1: Hasilkan data training baru
            logging.info("XGB Retrain: Menjalankan generate_training_data.py...")
            subprocess.run([python_executable, "generate_training_data.py"], check=True)

            # Langkah 2: Latih model baru
            logging.info("XGB Retrain: Menjalankan train_xgboost.py...")
            # Kita jalankan ini di latar belakang agar bot tidak berhenti
            xgb_retraining_process = subprocess.Popen([python_executable, "train_xgboost.py"])

            xgb_last_retrain_trade_count = current_trade_count
            logging.info("XGB Retrain: Proses pelatihan telah dimulai di latar belakang.")

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.error(f"XGB Retrain: Gagal memulai proses pelatihan ulang: {e}")

def reload_xgb_models_if_updated(config: Dict[str, Any], current_models: Dict[str, Any]):
    """Memeriksa model XGBoost yang baru dan memuatnya jika ada."""
    symbols = config.get('global_settings', {}).get('symbols_to_analyze', [])
    for symbol in symbols:
        model_path = f"xgboost_model_{symbol}.json"
        # Logika sederhana: periksa waktu modifikasi file
        # Implementasi yang lebih kuat akan menggunakan file penanda (.done)
        # atau memeriksa hash file.
        try:
            last_modified = datetime.fromtimestamp(os.path.getmtime(model_path))
            # Jika model diupdate dalam 10 menit terakhir, muat ulang
            if (datetime.now() - last_modified).total_seconds() < 600:
                 if model_path not in getattr(reload_xgb_models_if_updated, 'reloaded_once', []):
                    logging.info(f"--- Model XGBoost baru terdeteksi untuk {symbol}. Memuat ulang... ---")
                    model = xgb.XGBClassifier()
                    model.load_model(model_path)
                    current_models['xgb'][symbol] = model
                    logging.info(f"Model XGBoost untuk {symbol} berhasil diperbarui.")

                    # Tandai sebagai sudah di-reload untuk sesi ini
                    if not hasattr(reload_xgb_models_if_updated, 'reloaded_once'):
                        reload_xgb_models_if_updated.reloaded_once = []
                    reload_xgb_models_if_updated.reloaded_once.append(model_path)

        except (FileNotFoundError, OSError):
            continue # File mungkin belum dibuat

def main():
    """Fungsi utama untuk menjalankan bot."""
    global active_signals, signal_cooldown, gng_last_retrain_time, xgb_retraining_process, xgb_last_retrain_trade_count
    
    config = load_config()
    symbols = config.get('global_settings', {}).get('symbols_to_analyze', [])
    
    active_signals = {symbol: {} for symbol in symbols}
    signal_cooldown = {}

    gng_models, gng_stats, xgb_models = initialize_models(config)
    all_models = {
        'gng': gng_models,
        'gng_stats': gng_stats,
        'xgb': xgb_models
    }

    # Inisialisasi waktu retrain GNG
    for tf in gng_models.keys():
        gng_last_retrain_time[tf] = datetime.now()

    # Inisialisasi penghitung trade untuk retrain XGB
    try:
        with open(config.get("learning", {}).get("feedback_file", "trade_feedback.json"), 'r') as f:
            xgb_last_retrain_trade_count = len(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        xgb_last_retrain_trade_count = 0


    logging.info("="*50)
    logging.info("Bot Trading AI v5.2 (XGBoost & GNG Live Learning) Siap Beraksi!")
    logging.info("="*50)

    try:
        while True:
            # --- SIKLUS BELAJAR ADAPTIF (BOBOT) ---
            adapted_weights_per_profile = analyze_and_adapt_profiles(config)

            # --- SIKLUS BELAJAR GNG (STRUKTUR PASAR) ---
            retrain_gng_models_if_needed(config, all_models['gng'])

            # --- SIKLUS BELAJAR XGBOOST (VALIDATOR) ---
            trigger_xgb_retraining_if_needed(config)
            reload_xgb_models_if_updated(config, all_models)
            
            logging.info("--- Memulai Siklus Analisis Baru ---")
            
            for profile_name, profile_config in config.get("strategy_profiles", {}).items():
                if profile_config.get("enabled", False):
                    adapted_weights = adapted_weights_per_profile.get(profile_name, config.get("base_weights", {}))
                    process_profile(profile_name, config, all_models, adapted_weights)
            
            sleep_duration = config.get('global_settings', {}).get('main_loop_sleep_seconds', 20)
            logging.info("Semua profil telah dianalisis. Istirahat %d detik...", sleep_duration)
            time.sleep(sleep_duration)

    except KeyboardInterrupt:
        logging.info("Perintah berhenti diterima. Bot akan dimatikan.")
    finally:
        logging.info("Aplikasi Selesai.")

if __name__ == '__main__':
    main()
