# main.py (Refactored for Profiles & Learning)
#
# Deskripsi:
# Versi ini mendukung "Strategy Profiles" dari config.json dan
# mengimplementasikan siklus belajar adaptif untuk setiap profil.

import logging
import os
import time
import numpy as np
import xgboost as xgb
from datetime import datetime
from typing import Dict, Any
import json

from data_fetching import DataCache
from log_handler import WebServerHandler
from learning import analyze_and_adapt_profiles
from signal_generator import (
    analyze_tf_opportunity,
    build_signal_format,
    make_signal_id
)
from server_comm import send_signal_to_server
from agent import RuleBasedAgent

# --- Global States ---
DATA_CACHE = DataCache()
active_signals: Dict[str, Dict[str, Dict[str, any]]] = {}
signal_cooldown: Dict[str, datetime] = {}

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

def initialize_models(config: Dict[str, Any]) -> dict:
    """Inisialisasi semua model XGBoost untuk semua simbol."""
    xgb_models = {}
    global_conf = config.get('global_settings', {})
    symbols = global_conf.get('symbols_to_analyze', [])
    
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
            
    return xgb_models

def handle_opportunity(opp: Dict[str, Any], symbol: str, tf: str, config: Dict[str, Any], xgb_model: xgb.XGBClassifier, profile_name: str, agent: RuleBasedAgent):
    """Memproses, memvalidasi, dan mengirim sinyal jika ada peluang yang memenuhi syarat."""
    global active_signals, signal_cooldown
    
    profile_config = config['strategy_profiles'][profile_name]
    global_config = config['global_settings']

    # --- PENGEMBANGAN: Gunakan Agent untuk membuat keputusan ---
    decision = agent.decide(opp)
    
    if decision == "REJECT":
        return False

    logging.info("✅ [%s|%s|%s] SINYAL DITERIMA AGENT! Peluang %s. Skor: %.2f.", profile_name, symbol, tf, opp['signal'], opp['score'])
    
    if xgb_model and opp.get('features') is not None and opp['features'].size > 0:
        logging.info("[Arshy | %s|%s|%s] Meminta validasi kuantitatif dari Catelya...", profile_name, symbol, tf)
        features = np.array(opp['features']).reshape(1, -1)
        win_probability = xgb_model.predict_proba(features)[0][1]
        
        if win_probability < config.get('strategy_profiles', {}).get(profile_name, {}).get('xgboost_confidence_threshold', 0.75):
            logging.warning("[Catelya | %s|%s|%s] Probabilitas keberhasilan rendah (%.2f%%). Rekomendasi: BATALKAN.", profile_name, symbol, tf, win_probability * 100)
            return False
        
        logging.info("[Catelya | %s|%s|%s] Probabilitas keberhasilan terhitung: %.2f%%. Rekomendasi: LANJUTKAN.", profile_name, symbol, tf, win_probability * 100)

    # --- Persiapan & Pengiriman Sinyal ---
    order_type_to_use = opp.get('order_type', opp['signal'])
    signal_json = build_signal_format(
        symbol=symbol, entry_price=float(opp['entry_price_chosen']), 
        direction=opp['signal'], sl=float(opp['sl']), tp=float(opp['tp']), 
        order_type=order_type_to_use
    )
    
    # Payload sekarang menyertakan semua konteks untuk pembelajaran
    api_key = os.environ.get('API_KEY')
    secret_key = os.environ.get('SECRET_KEY')
    if not api_key or not secret_key:
        logging.error("API_KEY atau SECRET_KEY tidak ditemukan di environment variables. Sinyal tidak bisa dikirim.")
        return False

    payload = {
        "symbol": symbol,
        "signal_json": signal_json,
        "api_key": api_key,
        "server_url": global_config['server_url'],
        "secret_key": secret_key,
        "order_type": order_type_to_use,
        "score": opp.get('score'),
        "info": opp.get('info'),
        "profile_name": profile_name,
        "score_components": opp.get('score_components')
    }
    
    send_status = send_signal_to_server(**payload)

    if send_status == 'SUCCESS':
        sig_id = make_signal_id(signal_json)
        active_signals[symbol][sig_id] = {'signal_json': signal_json, 'tf': tf, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        logging.info("[%s|%s|%s] Sinyal berhasil dikirim!", profile_name, symbol, tf)
        signal_cooldown[symbol] = datetime.now()
        return True
        
    return False

def process_profile(profile_name: str, config: Dict[str, Any], models: Dict[str, Any], adapted_weights: Dict[str, float], agent: RuleBasedAgent):
    """Menjalankan seluruh siklus analisis untuk satu profil strategi."""
    global signal_cooldown
    
    profile_config = config['strategy_profiles'][profile_name]
    global_config = config['global_settings']
    
    # --- PENGEMBANGAN: Cek jam aktif untuk profil tertentu ---
    active_hours = profile_config.get('active_hours_utc')
    if active_hours:
        current_utc_hour = datetime.utcnow().hour
        is_active = False
        for hour_range in active_hours:
            try:
                start, end = map(int, hour_range.split('-'))
                if start <= current_utc_hour < end:
                    is_active = True
                    break
            except ValueError:
                logging.warning("Format 'active_hours_utc' salah di config untuk profil '%s'. Seharusnya 'HH-HH'. Contoh: ['08-10', '14-16']", profile_name)
                continue

        if not is_active:
            logging.info("Profil '%s' tidak aktif pada jam ini (UTC %d). Dilewati.", profile_name, current_utc_hour)
            return

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
                opp = analyze_tf_opportunity(
                    symbol=symbol, tf=tf, mt5_path=global_config['mt5_terminal_path'],
                    confidence_threshold=0.0,
                    min_distance_pips_per_tf=profile_config['min_distance_pips_per_tf'],
                    weights=adapted_weights, # Gunakan bobot yang sudah diadaptasi
                    indicator_settings=config.get('indicator_settings', {})
                )
                if opp and opp.get('signal') != "WAIT":
                    signal_sent = handle_opportunity(opp, symbol, tf, config, models['xgb'].get(symbol), profile_name, agent)
                    if signal_sent:
                        break 
            except Exception as e:
                logging.error("Error saat menganalisis %s|%s|%s: %s", profile_name, symbol, tf, e, exc_info=True)

def main():
    """Fungsi utama untuk menjalankan bot."""
    global active_signals, signal_cooldown
    
    config = load_config()
    symbols = config.get('global_settings', {}).get('symbols_to_analyze', [])
    
    active_signals = {symbol: {} for symbol in symbols}
    signal_cooldown = {}

    xgb_models = initialize_models(config)
    all_models = {'xgb': xgb_models}

    # --- PENGEMBANGAN: Inisialisasi Agent ---
    # Di masa depan, agent bisa dipilih berdasarkan konfigurasi
    agent = RuleBasedAgent(confidence_threshold=5.0) # Threshold bisa diambil dari config
    logging.info("Menggunakan Agent: %s", agent.name)

    logging.info("="*50)
    logging.info("Bot Trading AI v4.0 (Profile & Learning Enabled) Siap Beraksi!")
    logging.info("="*50)

    try:
        while True:
            # --- SIKLUS BELAJAR ---
            adapted_weights_per_profile = analyze_and_adapt_profiles(config)
            
            logging.info("--- Memulai Siklus Analisis Baru ---")
            
            for profile_name, profile_config in config.get("strategy_profiles", {}).items():
                if profile_config.get("enabled", False):
                    # Re-inisialisasi agent untuk setiap profil agar menggunakan threshold yang tepat
                    agent = RuleBasedAgent(confidence_threshold=profile_config.get('confidence_threshold', 5.0))

                    adapted_weights = adapted_weights_per_profile.get(profile_name, config.get("base_weights", {}))
                    process_profile(profile_name, config, all_models, adapted_weights, agent)
            
            sleep_duration = config.get('global_settings', {}).get('main_loop_sleep_seconds', 20)
            logging.info("Semua profil telah dianalisis. Istirahat %d detik...", sleep_duration)
            time.sleep(sleep_duration)

    except KeyboardInterrupt:
        logging.info("Perintah berhenti diterima. Bot akan dimatikan.")
    finally:
        logging.info("Aplikasi Selesai.")

if __name__ == '__main__':
    main()
