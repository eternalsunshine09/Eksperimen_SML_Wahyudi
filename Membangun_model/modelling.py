import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Deteksi apakah berjalan di GitHub Actions
is_github_actions = os.getenv('GITHUB_ACTIONS') == 'true'

# Konfigurasi MLflow
if not is_github_actions:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Telco_Churn_CI_Experiment")
mlflow.sklearn.autolog()

# Path data
data_path = "telco_churn_clean.csv" 
df = pd.read_csv(data_path)

# Persiapan Data
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Memulai pelatihan model...")
with mlflow.start_run(run_name="CI_Run"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model berhasil dilatih dengan akurasi: {acc:.4f}")

print("Pelatihan selesai.")