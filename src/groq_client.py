"""
Wrapper delgado sobre la API de Groq.

Deliberadamente NO usa LangChain ni ningún framework de orquestación:
llamadas directas al SDK de Groq para que sea fácil de leer, depurar y
explicar en el informe final (ver ajuste de arquitectura por nivel de
experiencia en el plan del reto).

Cubre dos cosas:
  - chat_completion(): el LLM que razona (Llama 3.1 70B) -> esto es lo que
    declaras en el informe para la compuerta G3.
  - transcribe_audio(): STT con Whisper Large V3. No es el modelo que
    "razona", así que no cuenta para la restricción de G3, pero corre en
    el mismo proveedor (Groq) para minimizar latencia acumulada.

Todas las llamadas devuelven también metadatos de uso (tokens, tiempo)
porque el README del reto exige reportar latencia P50/P95 y consumo de
tokens por turno/llamada. Instrumentar desde aquí evita tener que
retro-inyectar logging más adelante.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.1-70b-versatile")
STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY no está configurada. Copia .env.example a .env y "
        "completa tu API key (https://console.groq.com/)."
    )

_client = Groq(api_key=GROQ_API_KEY)


@dataclass
class ChatResult:
    """Resultado de una llamada de chat, con metadatos para instrumentación."""

    text: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str = LLM_MODEL
    raw: object = field(default=None, repr=False)


@dataclass
class TranscriptionResult:
    """Resultado de una transcripción STT, con metadatos para instrumentación."""

    text: str
    latency_seconds: float
    model: str = STT_MODEL
    raw: object = field(default=None, repr=False)


def chat_completion(
    messages: list[dict],
    model: str = LLM_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 512,
    response_format: dict | None = None,
) -> ChatResult:
    """
    Llama al LLM de razonamiento del agente.

    `messages` sigue el formato estándar OpenAI-like:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    `response_format` puedes pasarlo como {"type": "json_object"} cuando
    necesites la salida estructurada para la lógica de decisión
    (requiere_alerta / razon / documento_fuente, etc.).
    """
    start = time.perf_counter()
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if response_format is not None:
        kwargs["response_format"] = response_format

    completion = _client.chat.completions.create(**kwargs)
    elapsed = time.perf_counter() - start

    usage = getattr(completion, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    output_tokens = getattr(usage, "completion_tokens", None) if usage else None

    return ChatResult(
        text=completion.choices[0].message.content,
        latency_seconds=elapsed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        raw=completion,
    )


def transcribe_audio(
    audio_path: str | Path,
    model: str = STT_MODEL,
    language: str = "es",
) -> TranscriptionResult:
    """
    Transcribe un archivo de audio (turno del paciente) a texto.

    Pensado para interacción por turnos: se le pasa el audio completo de
    un turno ya grabado, no un stream continuo (ver ajuste de arquitectura
    en el plan del reto).
    """
    audio_path = Path(audio_path)
    start = time.perf_counter()
    with open(audio_path, "rb") as f:
        transcription = _client.audio.transcriptions.create(
            file=(audio_path.name, f.read()),
            model=model,
            language=language,
        )
    elapsed = time.perf_counter() - start

    return TranscriptionResult(
        text=transcription.text,
        latency_seconds=elapsed,
        model=model,
        raw=transcription,
    )


if __name__ == "__main__":
    # Prueba mínima de conectividad. Ejecuta:
    #   python -m src.groq_client
    # desde la raíz del proyecto, con GROQ_API_KEY ya configurada.
    result = chat_completion(
        messages=[
            {"role": "system", "content": "Eres un asistente de prueba. Responde en una frase corta."},
            {"role": "user", "content": "Saluda y confirma que la conexión con Groq funciona."},
        ]
    )
    print(f"[modelo={result.model}] [latencia={result.latency_seconds:.2f}s]")
    print(result.text)
