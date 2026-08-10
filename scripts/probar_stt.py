from pathlib import Path
from src.groq_client import transcribe_audio

AUDIO_PATH = Path("data/audio_pruebas/prueba_paciente.m4a")

if __name__ == "__main__":
    if not AUDIO_PATH.exists():
        print(f"⚠️  No encontré el archivo en {AUDIO_PATH} — verifica el nombre exacto.")
    else:
        resultado = transcribe_audio(AUDIO_PATH)
        print(f"Transcripción: {resultado.text}")
        print(f"Latencia: {resultado.latency_seconds:.2f}s")
        print(f"Modelo: {resultado.model}")