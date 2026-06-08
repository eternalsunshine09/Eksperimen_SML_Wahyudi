import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

# 1. Konfigurasi MLflow (Lokal)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Telco_Churn_Tuning_Experiment")

# 2. Memuat data dari folder yang sama
df = pd.read_csv("Membangun_model/telco_churn_clean.csv")

# 3. Persiapan Data
X = df.drop('Churn', axis=1) 
y = df['Churn']              
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Inisialisasi Model & Hyperparameter Tuning (Syarat Skilled)
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}

# 5. Memulai Eksperimen (Manual Logging)
print("Memulai proses Hyperparameter Tuning dengan GridSearchCV...")
with mlflow.start_run(run_name="Random_Forest_Tuning_Manual_Log"):
    
    # Menjalankan pencarian parameter terbaik
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # ---------------------------------------------------------
    # A. MANUAL LOGGING: Parameters (Syarat Skilled)
    # ---------------------------------------------------------
    mlflow.log_params(grid_search.best_params_)
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # ---------------------------------------------------------
    # B. MANUAL LOGGING: Metrics (Syarat Skilled)
    # ---------------------------------------------------------
    predictions = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    print(f"Akurasi Model Tuning: {acc:.4f}")
    
    # ---------------------------------------------------------
    # C. MANUAL LOGGING: Model & Artifacts
    # ---------------------------------------------------------
    signature = infer_signature(X_train, best_model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature
    )

print("Pelatihan dengan Manual Logging selesai. Silakan cek MLflow UI.")