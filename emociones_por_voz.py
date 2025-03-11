import streamlit as st
import speech_recognition as sr
import librosa
import numpy as np
import joblib

# Cargar el modelo de detección de emociones (previamente entrenado)
modelo_emociones = joblib.load("modelo_emociones.pkl")

# Función para extraer características de audio
def extraer_caracteristicas(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Configurar la aplicación Streamlit
st.title("🌊 EmoFluir - Detección de Emociones por Voz")
st.write("Habla durante unos segundos y descubre qué emoción predomina en tu voz.")

# Grabación de audio
recognizer = sr.Recognizer()
with sr.Microphone() as source:
    st.write("Presiona el botón y habla...")
    audio_data = recognizer.listen(source, timeout=5)
    st.write("Procesando audio...")
    
    # Guardar el audio en un archivo temporal
    with open("audio_temp.wav", "wb") as f:
        f.write(audio_data.get_wav_data())
    
    # Extraer características y predecir emoción
    caracteristicas = extraer_caracteristicas("audio_temp.wav")
    emocion_predicha = modelo_emociones.predict([caracteristicas])[0]
    
    st.write(f"🔍 **Emoción detectada:** {emocion_predicha}")
    
    # Recomendaciones según la emoción detectada
    recomendaciones = {
        "Tristeza": "💧 Bebe agua ionizada y realiza una meditación con agua para liberar emociones.",
        "Ansiedad": "🌊 Respira profundamente y sumérgete en un baño relajante con agua tibia.",
        "Estrés": "💦 Prueba la técnica del agua en movimiento: observa cómo fluye para calmar la mente.",
        "Alegría": "✨ Comparte tu energía positiva con alguien y agradece el presente."
    }
    
    st.write(f"🌀 **Recomendación:** {recomendaciones.get(emocion_predicha, 'Conéctate con el agua y escucha tu cuerpo.')} ")
