from dotenv import load_dotenv
import os

load_dotenv()  # Carga tus variables desde .env

# Recuperar el token
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
import whisperx
import torch
import gradio as gr
import os
from datetime import datetime

# 🔐 Cargar token desde variable de entorno
HF_TOKEN = os.getenv("HF_TOKEN")

# 🖥️ Detectar si hay GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 🎙️ Función principal para procesar audio
def transcribir(audio_path):
    # Cargar modelo de transcripción
    model = whisperx.load_model("medium", device=device, compute_type="float32", language="es")
    transcription = model.transcribe(audio_path)

    # Alinear palabras
    model_a, metadata = whisperx.load_align_model(language_code="es", device=device)
    aligned = whisperx.align(transcription["segments"], model_a, metadata, audio_path, device)

    # Diarización
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN)
    diarize_segments = diarize_model(audio_path)

    # Fusión manual
    from copy import deepcopy
    segments = deepcopy(aligned["segments"])
    for segment in segments:
        for diar in diarize_segments["segments"]:
            if diar["start"] <= segment["start"] < diar["end"]:
                segment["speaker"] = diar["speaker"]
                break

    # Formar texto
    full_text = "\n".join([f"[{seg.get('speaker', 'SPEAKER_X')}] {seg['text']}" for seg in segments])
    timestamp = datetime.now().strftime("Grabación %Y-%m-%d a las %H.%M.%S")
    return f"🕒 {timestamp}\n\n" + full_text

# 🎛️ Interfaz de usuario
ui = gr.Interface(
    fn=transcribir,
    inputs=gr.Audio(type="filepath", label="Sube tu audio"),
    outputs=gr.Textbox(label="Transcripción con Diarización"),
    title="🧠 Transcriptor WhisperX + Diarización",
    description="Sube un audio y obtén la transcripción con detección de voces."
)

# 🚀 Lanzar app
if __name__ == "__main__":
    ui.launch()
