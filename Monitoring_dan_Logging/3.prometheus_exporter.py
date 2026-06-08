from prometheus_client import start_http_server, Summary, Counter, Gauge
import time
import random

# Definisi 5 Metrik (Sesuai syarat Skilled)
REQUESTS = Counter('model_requests_total', 'Total prediksi')
ERRORS = Counter('model_errors_total', 'Total error')
LATENCY = Summary('prediction_latency_seconds', 'Waktu respon')
ACCURACY = Gauge('model_accuracy', 'Akurasi model')
CPU_USAGE = Gauge('system_cpu_usage', 'Persentase CPU')

def process_request():
    REQUESTS.inc()
    with LATENCY.time():
        time.sleep(0.1) # Simulasi latensi
    ACCURACY.set(0.79) # Sesuaikan dengan akurasi model Anda
    CPU_USAGE.set(random.uniform(10, 50))

if __name__ == '__main__':
    # Start server untuk dipantau Prometheus di port 8000
    start_http_server(8000)
    print("Exporter jalan di port 8000")
    while True:
        process_request()
        time.sleep(5)