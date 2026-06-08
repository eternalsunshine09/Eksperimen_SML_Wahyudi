import requests
import json

# 1. Tentukan URL endpoint API FastAPI Anda
URL = "http://localhost:8000/predict"

# 2. Siapkan data dummy
data_input = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

def run_inference():
    print(f"Mengirim request ke {URL}...")
    print(f"Data input: {data_input}\n")
    
    try:
        # 3. Mengirim request GET ke API
        response = requests.get(URL, json=data_input)
        
        # 4. Mengecek status dan menampilkan hasil prediksi
        if response.status_code == 200:
            print("Request Berhasil!")
            # Baris ini yang tadi hilang/error, sekarang sudah aman menggunakan .text
            print("Hasil Prediksi:", response.text) 
        else:
            print(f"Gagal. Status Code: {response.status_code}")
            print("Pesan Error:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: Tidak dapat terhubung ke API. Pastikan server FastAPI sudah menyala!")

if __name__ == "__main__":
    run_inference()