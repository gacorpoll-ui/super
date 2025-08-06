"""
train_neural_agent.py
=====================

Skrip untuk melatih "otak" Jaringan Syaraf Tiruan (Neural Network)
menggunakan data feedback trading. Model yang sudah dilatih akan disimpan
dan bisa digunakan oleh NeuralAgent.
"""

import json
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.preprocessing import StandardScaler

# --- Konfigurasi Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_nn_agent(feedback_file="trade_feedback_generated.json", model_output_path="neural_agent_model.pkl"):
    """
    Memuat data, melatih model MLPClassifier, dan menyimpannya.
    """
    logging.info(f"--- Memulai Pelatihan Agent Neural Network dari {feedback_file} ---")

    # --- 1. Memuat dan Mempersiapkan Data ---
    try:
        with open(feedback_file, "r") as f:
            trade_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Gagal memuat '{feedback_file}': {e}.")
        return

    # Filter data yang memiliki fitur
    valid_data = [d for d in trade_data if d.get('gng_input_features_on_signal')]
    if not valid_data:
        logging.error("Tidak ada data valid dengan fitur yang ditemukan di file feedback.")
        return

    features = [record['gng_input_features_on_signal'] for record in valid_data]
    labels = [1 if record['result'] == "WIN" else 0 for record in valid_data]
    X = np.array(features)
    y = np.array(labels)

    if len(X) < 50:
        logging.warning(f"Data latih tidak mencukupi ({len(X)} records). Pelatihan dibatalkan.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    logging.info(f"Data dibagi menjadi {len(X_train)} data latih dan {len(X_test)} data uji.")

    # --- 2. Scaling Fitur ---
    # Sangat penting untuk Neural Networks
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logging.info("Fitur telah di-scale menggunakan StandardScaler.")

    # --- 3. Mendefinisikan dan Melatih Model ---
    # Arsitektur sederhana: 2 hidden layer dengan 50 dan 25 neuron
    logging.info("Mendefinisikan model MLPClassifier...")
    nn_agent = MLPClassifier(
        hidden_layer_sizes=(50, 25),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=20
    )

    logging.info("Memulai pelatihan model...")
    nn_agent.fit(X_train_scaled, y_train)
    logging.info("Pelatihan model selesai.")

    # --- 4. Mengevaluasi Kinerja ---
    y_pred = nn_agent.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)

    logging.info(f"--- Hasil Evaluasi Agent Neural Network ---")
    logging.info(f"Akurasi   : {accuracy:.2%}")
    logging.info(f"Presisi   : {precision:.2%}")
    logging.info("\nLaporan Klasifikasi:\n" + classification_report(y_test, y_pred))
    logging.info("-------------------------------------------")

    # --- 5. Menyimpan Model dan Scaler ---
    # Penting untuk menyimpan scaler agar data baru bisa diproses dengan cara yang sama
    model_payload = {
        'model': nn_agent,
        'scaler': scaler
    }
    joblib.dump(model_payload, model_output_path)
    logging.info(f"Model agent dan scaler telah disimpan ke '{model_output_path}'.")
    logging.info("Anda sekarang bisa menggunakan model ini dengan NeuralAgent di main.py.")

if __name__ == '__main__':
    train_nn_agent()
