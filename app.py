import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import joblib
import cv2
import math
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
import gdown

# ==========================================
# 1. KONFIGURASI HALAMAN & CLEANING
# ==========================================
st.set_page_config(page_title="Dashboard AI Terintegrasi", page_icon="🤖", layout="wide")
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

current_dir = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 2. DEFINISI UNTUK KLASIFIKASI SAMPAH
# ==========================================
CLASS_NAMES = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

MODELS_DB = {
    "CNN (MobileNetV2)": {
        "type": "keras", "file": "model_sampah_terbaik.keras",
        "grafik": "grafik_training.png", "cm": "confusion_matrix.png",
        "stats": {"train_acc": 90.58, "val_acc": 88.61, "test_acc": 88.33}
    },
    "Backpropagation (MLP)": {
        "type": "keras", "file": "model_backpro_final.keras",
        "grafik": "grafik_backpro.png", "cm": "cm_backpro.png",
        "stats": {"train_acc": 69.37, "val_acc": 56.94, "test_acc": 53.89}
    },
    "SVM (Support Vector Machine)": {
        "type": "sklearn", "file": "model_svm.pkl",
        "grafik": None, "cm": "cm_svm.png",
        "stats": {"train_acc": 76.98, "val_acc": 58.68, "test_acc": 56.11}
    }
}

# ==========================================
# 3. UTILITIES: LOADER MODEL
# ==========================================
DRIVE_IDS = {
    "model_sampah_terbaik.keras": None,    # (Opsional jika CNN mau ditaruh drive juga)
    "model_backpro_final.keras":  "13T4PRUlSnM5MKpzcN6lbcVa61KBma5Uv", # <--- WAJIB ISI INI
    "model_svm.pkl":              "1r2pOm0zk0BkAXjbQTZik0oYigYS7h4k1" # SVM biasanya kecil, tidak perlu Drive
}

@st.cache_resource
def load_classification_model(model_name):
    try:
        model_info = MODELS_DB[model_name]
        filename = model_info['file']
        file_path = os.path.join(current_dir, filename)
        
        # --- LOGIC BARU: DOWNLOAD DARI DRIVE JIKA TIDAK ADA ---
        if not os.path.exists(file_path):
            file_id = DRIVE_IDS.get(filename)
            if file_id:
                with st.spinner(f"📥 Sedang mendownload model {model_name} (300MB+), harap tunggu..."):
                    url = f'https://drive.google.com/uc?id={file_id}'
                    gdown.download(url, file_path, quiet=False)
            else:
                return None, None
        # ------------------------------------------------------
        
        # Load Model seperti biasa
        if model_info['type'] == 'keras':
            return tf.keras.models.load_model(file_path), 'keras'
        elif model_info['type'] == 'sklearn':
            return joblib.load(file_path), 'sklearn'
            
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None, None

@st.cache_resource
def load_lstm_model():
    # Load Model LSTM Udara
    path = os.path.join(current_dir, 'lstm_airquality_heavy.h5')
    if os.path.exists(path):
        return tf.keras.models.load_model(path, compile=False) 
    return None

# ==========================================
# 4. UTILITIES: PREPROCESSING (GAMBAR & CSV)
# ==========================================
def preprocess_image(image_input, model_type, is_batch=False):
    if not is_batch: img_array = np.array(image_input)
    else: img_array = image_input

    if model_type == 'keras':
        if not is_batch:
            img_resized = cv2.resize(img_array, (224, 224))
            if img_resized.shape[-1] == 4: img_resized = img_resized[..., :3]
            return np.expand_dims(img_resized / 255.0, axis=0)
        else: return img_array 
    elif model_type == 'sklearn':
        if not is_batch:
            img_resized = cv2.resize(img_array, (64, 64))
            if img_resized.shape[-1] == 4: img_resized = img_resized[..., :3]
            return np.expand_dims((img_resized / 255.0).flatten(), axis=0)
        else:
            processed_batch = []
            for img in img_array:
                img_small = cv2.resize(img, (64, 64))
                processed_batch.append(img_small.flatten())
            return np.array(processed_batch)

# Fungsi Preprocessing KHUSUS DATA UDARA (Sesuai Training Anda)
def preprocess_air_quality(df_raw):
    # 1. Cleaning
    df = df_raw.copy()
    # Hapus kolom kosong jika ada (sesuai logic training)
    df.dropna(how='all', axis=1, inplace=True)
    df.dropna(how='all', axis=0, inplace=True)
    if 'NMHC(GT)' in df.columns:
        df.drop(columns=['NMHC(GT)'], inplace=True)
    
    # Replace -200 jadi NaN dan drop
    df.replace(-200, np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Parsing Date Time (Penting untuk visualisasi)
    if 'Date' in df.columns and 'Time' in df.columns:
        df['Time'] = df['Time'].astype(str).str.replace('.', ':', regex=False)
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
        df.set_index('DateTime', inplace=True)
        df.drop(columns=['Date', 'Time'], inplace=True)
    
    return df

def create_sequences(data, target_idx, time_steps=24):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :])
        y.append(data[i + time_steps, target_idx])
    return np.array(X), np.array(y)

# ==========================================
# 5. SIDEBAR & NAVIGASI
# ==========================================
st.sidebar.title("🎛️ Panel Utama")

# Pilihan Kategori Besar
app_mode = st.sidebar.selectbox("Pilih Mode Aplikasi:", 
                                ["♻️ Klasifikasi Sampah (Vision)", "🔮 Prediksi Kualitas Udara (LSTM)"])

st.sidebar.divider()



# ==========================================
# MODE A: KLASIFIKASI SAMPAH (CODE LAMA)
# ==========================================
if app_mode == "♻️ Klasifikasi Sampah (Vision)":
    
    # ... (Bagian Sidebar Sampah) ...
    selected_model_name = st.sidebar.selectbox("Model Vision:", list(MODELS_DB.keys()))
    model, model_type = load_classification_model(selected_model_name)
    
    menu = st.sidebar.radio("Menu Vision:", ["🏠 Klasifikasi Satuan", "✅ Evaluasi Massal", "📈 Laporan Training"])
    
    if menu == "🏠 Klasifikasi Satuan":
        st.title(f"🤖 {selected_model_name}")
        uploaded_file = st.file_uploader("Upload Foto Sampah", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Foto Upload", width=300)
            if st.button("Identifikasi"):
                if model:
                    processed = preprocess_image(image, model_type)
                    if model_type == 'keras': 
                        pred = model.predict(processed)[0]
                    else: 
                        pred = model.predict_proba(processed)[0]
                    
                    idx = np.argmax(pred)
                    st.success(f"Hasil: **{CLASS_NAMES[idx].upper()}**")
                    # st.bar_chart(pd.DataFrame(pred*100, index=CLASS_NAMES, columns=['%']))

    elif menu == "✅ Evaluasi Massal":
        st.title("✅ Uji Coba Dataset Massal")
        st.markdown(f"Menguji **{selected_model_name}** menggunakan file dataset `.npy` secara Live.")

        col_up1, col_up2 = st.columns(2)
        with col_up1:
            x_file = st.file_uploader("Upload X (Images)", type=['npy'])
        with col_up2:
            y_file = st.file_uploader("Upload y (Labels)", type=['npy'])

        if x_file and y_file:
            X_data = np.load(x_file)
            y_data = np.load(y_file)
            
            st.success(f"Dataset dimuat: {len(X_data)} sampel.")
            
            # --- PREVIEW GAMBAR ---
            with st.expander("📸 Preview Isi Dataset (1 Gambar per Kelas)", expanded=False):
                y_indices = np.argmax(y_data, axis=1)
                unique_classes = np.unique(y_indices)
                cols = st.columns(6)
                for i, class_idx in enumerate(unique_classes):
                    try:
                        sample_idx = np.where(y_indices == class_idx)[0][0]
                        sample_img = X_data[sample_idx]
                        with cols[i % 6]:
                            st.image(sample_img, caption=CLASS_NAMES[class_idx], use_container_width=True)
                    except:
                        pass

            st.divider()
            
            if st.button(f"▶️ Jalankan Evaluasi ({selected_model_name})"):
                if model:
                    progress_bar = st.progress(0)
                    status = st.empty()
                    
                    status.text("Preprocessing & Predicting...")
                    progress_bar.progress(20)
                    
                    # Preprocess Batch (Penting untuk SVM agar di-resize ke 64)
                    X_processed = preprocess_image(X_data, model_type, is_batch=True)
                    
                    # Predict
                    if model_type == 'keras':
                        y_pred_probs = model.predict(X_processed, batch_size=32, verbose=0)
                    else:
                        y_pred_probs = model.predict_proba(X_processed)
                    
                    progress_bar.progress(80)
                    status.text("Menghitung Metrik...")
                    
                    y_pred_classes = np.argmax(y_pred_probs, axis=1)
                    y_true_classes = np.argmax(y_data, axis=1)
                    
                    # Hitung Akurasi
                    acc = np.mean(y_pred_classes == y_true_classes) * 100
                    
                    progress_bar.progress(100)
                    status.text("Selesai!")
                    
                    # Tampilkan Skor
                    col_res1, col_res2 = st.columns(2)
                    col_res1.metric("LIVE ACCURACY", f"{acc:.2f}%")
                    col_res2.metric("JUMLAH DATA", f"{len(X_data)}")
                    
                    # Confusion Matrix Plot
                    st.subheader("📊 Live Confusion Matrix")
                    cm = confusion_matrix(y_true_classes, y_pred_classes)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
                    ax.set_ylabel('Label Asli')
                    ax.set_xlabel('Prediksi Model')
                    st.pyplot(fig)
                    
            

    elif menu == "📈 Laporan Training":
        st.title("Laporan Training Model")
        active_assets = MODELS_DB[selected_model_name]
        st.write(active_assets['stats'])
        if active_assets['grafik']: st.image(os.path.join(current_dir, active_assets['grafik']))
        if active_assets['cm']: st.image(os.path.join(current_dir, active_assets['cm']))

# ==========================================
# MODE B: PREDIKSI KUALITAS UDARA (UPDATED: FORECASTING & DOWNLOAD)
# ==========================================
elif app_mode == "🔮 Prediksi Kualitas Udara (LSTM)":
    
    # --- SUB-MENU NAVIGASI ---
    menu_lstm = st.sidebar.radio(
        "Menu Evaluasi LSTM:",
        [
            "📈 Evaluasi Data Training (80%)", 
            "✅ Evaluasi Data Testing (20%)",
            "🚀 Prediksi Masa Depan (Forecasting)" # <-- MENU BARU
        ]
    )
    
    st.title(f"🔮 {menu_lstm}")
    
    # --- FITUR DOWNLOAD DATASET (Agar orang lain bisa coba) ---
    st.markdown("""
    **Langkah 1:** Download dataset sampel jika belum punya.
    **Langkah 2:** Upload file tersebut ke aplikasi.
    """)
    
    # Cek apakah file ada di server untuk didownload user
    csv_path = os.path.join(current_dir, 'AirQualityUCI.csv')
    if os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            st.download_button(
                label="📥 Download Dataset Sampel (AirQualityUCI.csv)",
                data=f,
                file_name="AirQualityUCI.csv",
                mime="text/csv",
                help="Klik untuk mengunduh data agar bisa Anda upload kembali untuk testing."
            )
    else:
        st.warning("⚠️ File 'AirQualityUCI.csv' tidak ditemukan di folder server. Pastikan file ada satu folder dengan app.py")

    st.divider()

    lstm_model = load_lstm_model()
    
    if lstm_model is None:
        st.error("❌ Model 'lstm_airquality_heavy.h5' tidak ditemukan di folder proyek!")
    else:
        # File Uploader
        uploaded_csv = st.file_uploader("Upload File CSV Data Udara", type=['csv'])
        
        if uploaded_csv:
            # 1. BACA & BERSIHKAN DATA
            try:
                df_raw = pd.read_csv(uploaded_csv, sep=';', decimal=',')
            except:
                uploaded_csv.seek(0)
                df_raw = pd.read_csv(uploaded_csv)
            
            df_clean = preprocess_air_quality(df_raw)
            
            # --- PENGATURAN UMUM ---
            dataset = df_clean.values
            target_col = 'CO(GT)'
            
            if target_col not in df_clean.columns:
                st.error(f"Kolom target '{target_col}' tidak ditemukan!")
            else:
                target_idx = list(df_clean.columns).index(target_col)
                
                # Scale Fitur & Target
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(dataset)
                
                scaler_target = MinMaxScaler(feature_range=(0, 1))
                scaler_target.fit(df_clean[[target_col]])
                
                LOOKBACK = 24 # Window input model

                # ====================================================
                # LOGIC A: EVALUASI (TRAINING & TESTING) - KODE LAMA
                # ====================================================
                if "Evaluasi" in menu_lstm:
                    st.subheader("⚙️ Pengaturan Evaluasi")
                    window_size = st.slider("Jangka Waktu Tampil (Jam):", 24, 1000, 200, 24)
                    
                    if st.button("▶️ Jalankan Evaluasi"):
                        with st.spinner("Memproses Data..."):
                            X, y = create_sequences(scaled_data, target_idx, time_steps=LOOKBACK)
                            train_size = int(len(X) * 0.8)
                            
                            if "Data Training" in menu_lstm:
                                X_eval = X[:train_size]
                                y_eval = y[:train_size]
                                label_grafik, line_color = "Data Training", '#1f77b4'
                            else:
                                X_eval = X[train_size:]
                                y_eval = y[train_size:]
                                label_grafik, line_color = "Data Testing", '#ff7f0e'
                            
                            # Limit window
                            if len(X_eval) > window_size:
                                X_eval, y_eval = X_eval[:window_size], y_eval[:window_size]

                            # Prediksi
                            predictions = lstm_model.predict(X_eval)
                            pred_inv = scaler_target.inverse_transform(predictions)
                            y_actual_inv = scaler_target.inverse_transform(y_eval.reshape(-1, 1))
                            
                            # Metrik
                            rmse = math.sqrt(mean_squared_error(y_actual_inv, pred_inv))
                            r2 = r2_score(y_actual_inv, pred_inv)
                            
                            col1, col2 = st.columns(2)
                            col1.metric("RMSE", f"{rmse:.4f}")
                            col2.metric("R2 Score", f"{r2:.4f}")
                            
                            # Grafik
                            chart_df = pd.DataFrame({'Aktual': y_actual_inv.flatten(), 'Prediksi': pred_inv.flatten()})
                            st.line_chart(chart_df, color=["#d3d3d3", line_color])

                # ====================================================
                # LOGIC B: PREDIKSI MASA DEPAN (FORECASTING) - BARU!
                # ====================================================
                elif "Masa Depan" in menu_lstm:
                    st.subheader("🚀 Peramalan Masa Depan (Forecasting)")
                    st.info("Model akan menggunakan 24 jam data TERAKHIR dari CSV untuk memprediksi jam-jam berikutnya.")
                    
                    # Slider Prediksi Masa Depan
                    future_steps = st.slider("Mau meramal berapa jam ke depan?", 
                                             min_value=1, max_value=168, value=24, step=1,
                                             help="Maksimal 1 minggu (168 jam). Semakin jauh, akurasi mungkin menurun.")
                    
                    if st.button("🔮 Mulai Meramal"):
                        with st.spinner(f"Sedang meramal {future_steps} jam ke depan..."):
                            
                            # 1. Ambil Sequence Terakhir dari Data (24 jam terakhir)
                            last_sequence = scaled_data[-LOOKBACK:]
                            current_sequence = last_sequence.copy() # (24, n_features)
                            
                            future_predictions = []
                            
                            # 2. Looping Prediksi Rekursif
                            # Kita prediksi jam ke-1, hasilnya dipakai buat prediksi jam ke-2, dst.
                            for _ in range(future_steps):
                                # Reshape biar bisa masuk LSTM (1, 24, n_features)
                                input_seq = current_sequence.reshape(1, LOOKBACK, current_sequence.shape[1])
                                
                                # Prediksi 1 langkah
                                pred = lstm_model.predict(input_seq, verbose=0) # Output (1, 1)
                                
                                # Simpan hasil (dalam skala 0-1)
                                future_predictions.append(pred[0, 0])
                                
                                # --- UPDATE SEQUENCE ---
                                # Kita butuh 'mock' data baru untuk fitur lain selain target.
                                # Karena kita tidak tahu suhu/kelembaban masa depan, kita asumsikan 
                                # fitur lain KONSTAN (pakai nilai jam terakhir) atau sama dengan nilai prediksi.
                                # Cara simpel: Copy baris terakhir, update kolom target dengan prediksi baru.
                                
                                new_row = current_sequence[-1].copy()
                                new_row[target_idx] = pred[0, 0] # Masukkan hasil prediksi ke kolom target
                                
                                # Buang jam terlama (index 0), masukkan jam baru (new_row)
                                current_sequence = np.append(current_sequence[1:], [new_row], axis=0)

                            # 3. Kembalikan ke Skala Asli
                            future_predictions_array = np.array(future_predictions).reshape(-1, 1)
                            future_inv = scaler_target.inverse_transform(future_predictions_array)
                            
                            # 4. Buat Tanggal Masa Depan untuk Grafik
                            last_date = df_clean.index[-1]
                            future_dates = pd.date_range(start=last_date, periods=future_steps + 1, freq='H')[1:]
                            
                            # 5. Visualisasi
                            # Gabungkan Data Lama (Misal 48 jam terakhir) + Data Masa Depan
                            show_history = 48
                            history_data = df_clean[target_col].iloc[-show_history:].values
                            history_dates = df_clean.index[-show_history:]
                            
                            # Plotting pakai Matplotlib agar bisa custom warna garis putus-putus
                            fig, ax = plt.subplots(figsize=(10, 5))
                            
                            # Garis Data Sejarah
                            ax.plot(history_dates, history_data, label='Data Historis (Terakhir)', color='blue')
                            
                            # Garis Prediksi Masa Depan
                            ax.plot(future_dates, future_inv, label='Ramalan Masa Depan', color='red', linestyle='--', marker='o', markersize=4)
                            
                            ax.set_title(f"Prediksi Kualitas Udara {future_steps} Jam Kedepan")
                            ax.set_ylabel("Konsentrasi CO (mg/m3)")
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            
                            # Rotate tanggal biar rapi
                            plt.xticks(rotation=45)
                            
                            st.pyplot(fig)
                            
                            # Tampilkan Tabel
                            st.success("Ramalan Selesai!")
                            with st.expander("Lihat Angka Prediksi"):
                                df_future = pd.DataFrame({
                                    'Waktu': future_dates,
                                    'Prediksi CO': future_inv.flatten()
                                })
                                st.dataframe(df_future)

st.sidebar.divider()

st.sidebar.caption("Project by:")

# Menggunakan Expander agar rapi (bisa diklik untuk melihat detail)
with st.sidebar.expander("👥 Anggota Kelompok", expanded=False):
    st.markdown("""
    **1. Irfan Ibrahim** NIM: 2201020100
    
    **2. Maulana Fitra Ramadhani** NIM: 2201020105
    
    **3. Muhammad Ridho** NIM: 2201020104
    
    **4. M. Wisnu Adjie Pramudya** NIM: 2201020109
    
    **5. Fiana Wahyu Laura** NIM: 2301020082
    """)


st.sidebar.divider()
