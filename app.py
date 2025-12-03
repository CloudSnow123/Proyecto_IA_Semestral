import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Configuración visual de la página
st.set_page_config(page_title="Detector de Emociones IA", page_icon="🎙️", layout="centered")

# Diccionario para traducir números a texto
EMOCIONES_TEXTO = {
    0: "Alegría 😄",
    1: "Disgusto 🤢", 
    2: "Enojo 😠",
    3: "Miedo 😨",
    4: "Neutro 😐",
    5: "Tristeza 😢",
}


# Cargar el modelo entrenado
@st.cache_resource
def cargar_ia():
    try:
        modelo = joblib.load('modelo_mlp.pkl')
        scaler = joblib.load('scaler.pkl')
        return modelo, scaler
    except FileNotFoundError:
        return None, None

model, scaler = cargar_ia()

# Título y Descripción
st.title("🎙️ Detección de Emociones en Audio")
st.write("Sube un audio en español y la Red Neuronal (MLP) analizará el tono de voz.")

# Verificar si el modelo existe
if model is None:
    st.error("⚠️ Error: No se encuentra el archivo 'modelo_mlp.pkl'. Primero ejecuta el archivo entrenamiento.py")
else:
    # Subida de archivo
    audio_file = st.file_uploader("Sube tu archivo .wav aquí", type=["wav"])

    if audio_file is not None:
        # 1. Mostrar reproductor de audio
        st.audio(audio_file, format='audio/wav')
        
        if st.button("🔍 Analizar Emoción"):
            with st.spinner('Escuchando y procesando...'):
                try:
                    # 2. Preprocesamiento (Igual que en el entrenamiento)
                    y, sr = librosa.load(audio_file, res_type='kaiser_fast')
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                    mfccs_promedio = np.mean(mfccs.T, axis=0)
                    
                    # Darle forma de matriz (1 fila, 40 columnas)
                    features = mfccs_promedio.reshape(1, -1)
                    
                    # 3. Escalar los datos
                    features_scaled = scaler.transform(features)
                    
                    # 4. Predicción
                    prediccion_idx = model.predict(features_scaled)[0]
                    probs = model.predict_proba(features_scaled)[0]
                    emocion_detectada = EMOCIONES_TEXTO.get(prediccion_idx, "Desconocido")
                    
                    # 5. Mostrar Resultados
                    st.success("¡Análisis completado!")
                    
                    # Tarjeta de resultado principal
                    st.markdown(f"""
                        <div style="text-align: center; padding: 20px; background-color: #262730; border-radius: 10px; margin-bottom: 20px;">
                            <h3 style="margin:0; color: #FAFAFA;">La emoción predominante es:</h3>
                            <h1 style="margin:0; font-size: 3em; color: #4CAF50;">{emocion_detectada}</h1>
                            <p style="color: #9E9E9E;">Confianza: {max(probs)*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # Barras de progreso detalladas
                    st.subheader("Desglose de probabilidades:")
                    for idx, prob in enumerate(probs):
                        nombre_emocion = EMOCIONES_TEXTO.get(idx, f"Clase {idx}")
                        st.write(f"**{nombre_emocion}**")
                        st.progress(float(prob))
                        
                except Exception as e:
                    st.error(f"Ocurrió un error al procesar el audio: {e}")
