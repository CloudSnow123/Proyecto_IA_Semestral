import librosa
import numpy as np
import os
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# CONFIGURACIÓN
# ==========================================
DATASET_PATH = "Dataset"  # Nombre de carpeta principal
EMOCIONES = {
    "Alegria":0,
    "Disgusto":1,
    "Enojo":2,
    "Miedo":3,
    "Neutro":4,
    "Tristeza":5,
}

# ==========================================
# 1. EXTRACCIÓN DE CARACTERÍSTICAS (Feature Extraction)
# ==========================================
def extract_mfcc(file_path):
    try:
        # Carga el audio
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # Esto convierte el audio en una matriz de números
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Promediamos para obtener un vector único por audio (input de la MLP)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error leyendo {file_path}: {e}")
        return None

# ==========================================
# 2. CARGA DEL DATASET
# ==========================================
print("📂 Cargando audios y procesando...")
X = [] # Características (Input)
y = [] # Etiquetas (Output)

for emocion, etiqueta in EMOCIONES.items():
    # Busca archivos .wav en cada subcarpeta
    search_path = os.path.join(DATASET_PATH, emocion, "*.wav")
    archivos = glob.glob(search_path)

    print(f"🔍 Intentando buscar en la ruta: {search_path}")
    
    print(f"   -> Procesando '{emocion}': {len(archivos)} archivos encontrados.")
    
    for file in archivos:
        features = extract_mfcc(file)
        if features is not None:
            X.append(features)
            y.append(etiqueta)

# Convertir a arrays de Numpy
X = np.array(X)
y = np.array(y)

# Verificar si hay datos
if len(X) == 0:
    print("❌ ERROR: No se encontraron audios. Revisa que la carpeta 'dataset' exista y tenga archivos .wav dentro.")
    exit()

# ==========================================
# 3. PREPARACIÓN Y ENTRENAMIENTO
# ==========================================
# Dividir datos: 80% para aprender, 20% para examen final
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalización (Vital para que la Red Neuronal converja)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n🧠 Iniciando entrenamiento de la Red Neuronal (MLP)...")

# Definición del Modelo MLP (Multi-Layer Perceptron)
# - hidden_layer_sizes: 2 capas ocultas con 256 y 128 neuronas
# - max_iter: 500 intentos de aprendizaje
model = MLPClassifier(
    hidden_layer_sizes=(256, 128), 
    activation='relu', 
    solver='adam', 
    max_iter=500, 
    batch_size=32,
    verbose=True  # Muestra el progreso en tiempo real
)

model.fit(X_train, y_train)

# ==========================================
# 4. EVALUACIÓN Y GUARDADO
# ==========================================
print("\n📊 Evaluando modelo...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Precisión final: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=EMOCIONES.keys()))

# Guardar el cerebro (modelo) y la escala (scaler)
joblib.dump(model, 'modelo_mlp.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("💾 Modelo guardado como 'modelo_mlp.pkl'. Listo para usar en la App.")