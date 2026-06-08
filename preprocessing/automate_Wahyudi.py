import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# 1. Fungsi untuk memuat data
def load_data(file_path):
    print(f"Memuat data dari: {file_path}")
    return pd.read_csv(file_path)

# 2. Fungsi utama untuk preprocessing (sama persis dengan di notebook)
def preprocess_data(df):
    print("Memulai proses pembersihan data...")
    
    # Memperbaiki TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    
    # Menghapus customerID
    df.drop('customerID', axis=1, inplace=True)
    
    # Mengonversi target Churn
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Encoding fitur kategorikal lainnya
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        
    print("Pemrosesan data selesai.")
    return df

# 3. Fungsi untuk menyimpan data yang sudah bersih
def save_data(df, output_path):
    # Membuat folder secara otomatis jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data bersih berhasil disimpan di: {output_path}")

# =====================================================================
# BLOK EKSEKUSI UTAMA
# =====================================================================
if __name__ == "__main__":
    
    # 1. Tentukan lokasi data mentah berada (Sudah disamakan dengan notebook)
    input_file = "WA_Fn-UseC_-Telco-Customer-Churn.csv" 
    
    # 2. Tentukan lokasi untuk menyimpan data bersihnya
    output_file = "dataset_preprocessing/telco_churn_clean.csv"
    
    # 3. Jalankan urutan prosesnya: Load -> Preprocess -> Save
    raw_data = load_data(input_file)
    clean_data = preprocess_data(raw_data)
    save_data(clean_data, output_file)