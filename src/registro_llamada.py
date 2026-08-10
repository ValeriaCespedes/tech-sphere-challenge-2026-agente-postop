"""
Persistencia de la llamada en curso: guarda cada turno (transcripción,
respuesta, fuentes RAG, clasificación de triaje) en un archivo JSON por
llamada. Sirve tanto para el resumen final como para las métricas del
README (latencias, tokens, consultas al RAG).
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

DIR_LLAMADAS = Path("data/llamadas")
DIR_LLAMADAS.mkdir(parents=True, exist_ok=True)


def nueva_llamada(nombre_paciente: str = "", procedimiento: str = "") -> str:
    """Crea un registro nuevo de llamada y devuelve su ID."""
    llamada_id = str(uuid.uuid4())[:8]
    registro = {
        "llamada_id": llamada_id,
        "inicio": datetime.now().isoformat(),
        "nombre_paciente": nombre_paciente or "No identificado",
        "procedimiento": procedimiento or "No especificado",
        "turnos": [],
    }
    _guardar(llamada_id, registro)
    return llamada_id

def agregar_turno(llamada_id: str, turno: dict) -> None:
    """Agrega un turno al registro de una llamada existente."""
    registro = _cargar(llamada_id)
    turno["timestamp"] = datetime.now().isoformat()
    registro["turnos"].append(turno)
    _guardar(llamada_id, registro)


def obtener_llamada(llamada_id: str) -> dict:
    return _cargar(llamada_id)


def _ruta(llamada_id: str) -> Path:
    return DIR_LLAMADAS / f"{llamada_id}.json"


def _guardar(llamada_id: str, registro: dict) -> None:
    _ruta(llamada_id).write_text(
        json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _cargar(llamada_id: str) -> dict:
    return json.loads(_ruta(llamada_id).read_text(encoding="utf-8"))