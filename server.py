"""
Servidor de la interfaz de llamada. Endpoints:
  POST /llamada/iniciar -> crea una llamada nueva, devuelve su ID
  POST /turno -> procesa un turno de audio dentro de una llamada existente
  GET /llamada/{id}/resumen -> genera el resumen estructurado de la llamada
"""

import io
import time
import tempfile
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.groq_client import transcribe_audio
from src.responder import responder_con_contexto
from src.tts import sintetizar
from src.decision import clasificar_turno
from src.registro_llamada import nueva_llamada, agregar_turno
from src.resumen import generar_resumen

app = FastAPI(title="Agente de seguimiento postoperatorio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DatosLlamada(BaseModel):
    nombre_paciente: str = ""
    procedimiento: str = ""


@app.post("/llamada/iniciar")
async def iniciar_llamada(datos: DatosLlamada):
    llamada_id = nueva_llamada(
        nombre_paciente=datos.nombre_paciente,
        procedimiento=datos.procedimiento,
    )
    return JSONResponse({"llamada_id": llamada_id})


@app.post("/turno")
async def procesar_turno(audio: UploadFile = File(...), llamada_id: str = Form(...)):
    inicio_total = time.perf_counter()

    audio_bytes = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        ruta_temporal = Path(tmp.name)

    # 1. STT
    resultado_stt = transcribe_audio(ruta_temporal)
    ruta_temporal.unlink(missing_ok=True)

    # 2. Clasificación de triaje
    decision = clasificar_turno(resultado_stt.text)

    # 3. RAG + LLM para la respuesta conversacional
    resultado_texto = responder_con_contexto(resultado_stt.text)

    # 4. Construye la respuesta final, incorporando la decisión de triaje
    respuesta_final = resultado_texto["respuesta"]
    if decision.requiere_mas_info and decision.pregunta_seguimiento:
        respuesta_final = decision.pregunta_seguimiento
    elif decision.clasificacion == "rojo":
        respuesta_final = (
            "Lo que me cuenta requiere atención médica inmediata. "
            "Voy a escalar esto a nuestro equipo médico ahora mismo y alguien se "
            "comunicará con usted a la brevedad. Si los síntomas empeoran, acuda "
            "a urgencias de inmediato."
        )
    elif decision.clasificacion == "amarillo":
        respuesta_final = (
            resultado_texto["respuesta"]
            + " Voy a dejar registrado su reporte para que el equipo médico le dé seguimiento."
        )

    # 5. TTS
    resultado_tts = sintetizar(respuesta_final)

    latencia_total = time.perf_counter() - inicio_total

    # 6. Persistencia del turno completo
    # Las fuentes solo cuentan como "usadas" si la respuesta final realmente
# fue la generada por el RAG. Si se sobrescribió con un mensaje de triaje
# genérico (rojo/amarillo), esas fuentes no fundamentan lo que se dijo.
    respuesta_fue_del_rag = (respuesta_final == resultado_texto["respuesta"])

    agregar_turno(llamada_id, {
        "transcripcion_paciente": resultado_stt.text,
        "clasificacion": decision.clasificacion,
        "razon_clasificacion": decision.razon,
        "fuente_decision": decision.fuente_decision,
        "requiere_mas_info": decision.requiere_mas_info,
        "respuesta_agente": respuesta_final,
        "fuentes_rag": resultado_texto["fuentes"] if respuesta_fue_del_rag else [],
        "sin_conocimiento": resultado_texto["sin_conocimiento"],
        "latencia_stt": resultado_stt.latency_seconds,
        "latencia_llm": resultado_texto.get("latencia"),
        "latencia_tts": resultado_tts.latency_seconds,
        "latencia_total": latencia_total,
        "tokens": resultado_texto.get("tokens"),
    })

    print(f"[TURNO {llamada_id}] {decision.clasificacion.upper()} - {resultado_stt.text[:60]}...")

    return StreamingResponse(
        io.BytesIO(resultado_tts.audio_bytes),
        media_type="audio/wav",
        headers={
            "X-Transcripcion": quote(resultado_stt.text),
            "X-Respuesta-Texto": quote(respuesta_final),
            "X-Clasificacion": decision.clasificacion,
        },
    )


@app.get("/llamada/{llamada_id}/resumen")
async def obtener_resumen(llamada_id: str):
    resumen = generar_resumen(llamada_id)
    return JSONResponse(resumen)


@app.get("/salud")
async def salud():
    return JSONResponse({"estado": "ok"})


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")