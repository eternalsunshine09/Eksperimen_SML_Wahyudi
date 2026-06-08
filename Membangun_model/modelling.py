import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Konfigurasi MLflow (Lokal)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Telco_Churn_Basic_Experiment")

# 2. SYARAT BASIC: Menggunakan autolog dari MLflow
mlflow.sklearn.autolog()

# 3. Memuat data bersih hasil Preprocessing
# Path menyesuaikan struktur file terakhir Anda
df = pd.read_csv("Membangun_model/telco_churn_clean.csv")

# 4. Persiapan Data (Split Training & Testing)
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. SYARAT BASIC: Melatih model tanpa hyperparameter tuning
print("Memulai pelatihan model dasar...")
with mlflow.start_run(run_name="Random_Forest_Basic_Run"):
    # Model dilatih menggunakan parameter bawaan saja
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluasi sederhana
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model berhasil dilatih dengan akurasi: {acc:.4f}")

print("Pelatihan selesai. Silakan cek MLflow UI.")