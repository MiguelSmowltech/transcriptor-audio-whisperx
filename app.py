import whisperx
import torch
import gradio as gr
from datetime import datetime
from dotenv import load_dotenv
import os

# 🛡️ Cargar token desde .env
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# 🪖 Detectar dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🤗 Cargar modelo WhisperX y de alineación
model = whisperx.load_model("medium", device=DEVICE, compute_type="float32", language="es")
model_a, metadata = whisperx.load_align_model(language_code="es", device=DEVICE)

# 🔊 Diarización de hablantes (requiere login Hugging Face)
diarize_model = whisperx.DiarizationPipeline(use_auth_token=HUGGINGFACE_TOKEN)

# 🌐 Palabras sospechosas (recortadas para ejemplo)
palabras_sospechosas = ["whatsapp", "telegram", "discord", "ayúdame", "copiar", "chuleta", "trampa"]

def evaluar_audio(audio_path):
    transcription = model.transcribe(audio_path)
    aligned = whisperx.align(transcription["segments"], model_a, metadata, audio_path, DEVICE)
    diarize_segments = diarize_model(audio_path)

    # ✨ Fusionar segmentos con speaker
    segments = aligned["segments"]
    for segment in segments:
        for diar in diarize_segments["segments"]:
            if diar["start"] <= segment["start"] < diar["end"]:
                segment["speaker"] = diar["speaker"]
                break

    full_text = " ".join([seg["text"] for seg in segments])

    # 🔎 Palabras no permitidas
    palabras_detectadas = [p for p in palabras_sospechosas if p.lower() in full_text.lower()]

    # 🗓️ Timestamp
    timestamp = datetime.now().strftime("Grabación %Y-%m-%d a las %H.%M.%S")

    # 📄 Generar informe
    resultado = f"Transcripción de la {timestamp}\n\n"
    resultado += full_text + "\n\n"
    if palabras_detectadas:
        resultado += f"Palabras no permitidas detectadas: {', '.join(palabras_detectadas)}"
    else:
        resultado += "No se detectaron palabras no permitidas."

    return resultado

# 🚀 Crear interfaz con Gradio
interfaz = gr.Interface(
    fn=evaluar_audio,
    inputs=gr.Audio(type="filepath", label="Sube tu archivo de audio (.mp3, .wav, etc.)"),
    outputs=gr.Textbox(label="Resultado de la transcripción y análisis"),
    title="🔊 Transcriptor con Análisis de Conductas Sospechosas",
    description="Sube un audio para transcribirlo, analizarlo y detectar palabras potencialmente sospechosas."
)

# 🌐 Lanzar app
if __name__ == "__main__":
    interfaz.launch()
