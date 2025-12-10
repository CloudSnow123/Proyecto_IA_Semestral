# 🎙️ Detector de Emociones en Audio con IA

> **Una aplicación web interactiva que utiliza Deep Learning para identificar emociones humanas a partir de la voz.**

Este proyecto implementa una **Red Neuronal Perceptrón Multicapa (MLP)** capaz de analizar archivos de audio en formato `.wav`, extraer características acústicas (MFCC) y clasificar la emoción del hablante en tiempo real a través de una interfaz amigable construida con **Streamlit**.

---

## ✨ Características Principales

* **Análisis de Audio:** Procesamiento digital de señales utilizando la librería `librosa`.
* **Interfaz Intuitiva:** Subida de archivos "Drag & Drop" y reproductor de audio integrado.
* **Clasificación de 6 Emociones:**
    * 😄 Alegría
    * 🤢 Disgusto
    * 😠 Enojo
    * 😨 Miedo
    * 😐 Neutro
    * 😢 Tristeza
* **Visualización de Datos:** Muestra la emoción predominante y un desglose porcentual de confianza para cada categoría.

---

## 🛠️ Stack Tecnológico

El proyecto fue construido utilizando las siguientes tecnologías:

* **Lenguaje:** [Python 3.10+](https://www.python.org/)
* **Frontend:** [Streamlit](https://streamlit.io/) (Framework para Data Apps)
* **Procesamiento de Audio:** Librosa, NumPy
* **Machine Learning:** Scikit-Learn (Entrenamiento), Joblib (Persistencia de modelos)
* **Visualización:** Matplotlib

---