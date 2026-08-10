"""
Wrapper sobre Kokoro TTS (español, voz ef_dora).

Convierte texto de respuesta del agente en audio listo para reproducir.
Instrumentado con latencia porque el README exige reportar tiempos del
pipeline completo, no solo del LLM.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field

import soundfile as sf
from kokoro import KPipeline

LANG_CODE = "e"
VOZ_DEFECTO = "ef_dora"
SAMPLE_RATE = 24000

_pipeline = KPipeline(lang_code=LANG_CODE)


@dataclass
class SintesisResult:
    audio_bytes: bytes
    latency_seconds: float
    voz: str = VOZ_DEFECTO
    raw: object = field(default=None, repr=False)


def sintetizar(texto: str, voz: str = VOZ_DEFECTO) -> SintesisResult:
    """
    Convierte texto en audio (WAV, 24kHz) y lo devuelve como bytes en memoria,
    listo para enviar por HTTP sin escribir a disco.
    """
    start = time.perf_counter()
    generator = _pipeline(texto, voice=voz)

    # Kokoro puede devolver el audio en varios fragmentos si el texto es largo;
    # los concatenamos en un solo array antes de codificar a WAV.
    fragmentos = [audio for _, _, audio in generator]

    import numpy as np
    audio_completo = np.concatenate(fragmentos) if len(fragmentos) > 1 else fragmentos[0]

    buffer = io.BytesIO()
    sf.write(buffer, audio_completo, SAMPLE_RATE, format="WAV")
    audio_bytes = buffer.getvalue()

    elapsed = time.perf_counter() - start

    return SintesisResult(
        audio_bytes=audio_bytes,
        latency_seconds=elapsed,
        voz=voz,
    )


if __name__ == "__main__":
    # Prueba mínima: python -m src.tts
    resultado = sintetizar("Buenos días, soy su asistente de seguimiento postoperatorio.")
    with open("data/prueba_tts_wrapper.wav", "wb") as f:
        f.write(resultado.audio_bytes)
    print(f"Latencia: {resultado.latency_seconds:.2f}s")
    print("Guardado en data/prueba_tts_wrapper.wav")