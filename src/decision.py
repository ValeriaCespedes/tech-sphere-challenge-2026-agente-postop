"""
Lógica de decisión de escalamiento: clasifica cada turno del paciente en
verde/amarillo/rojo, combinando salida estructurada del LLM con una capa
de reglas duras como red de seguridad.

Asimetría clínica: un falso negativo (no alertar cuando debía) es mucho
peor que un falso positivo. Ante la duda, la decisión final escala hacia
el nivel más alto detectado entre LLM y reglas, nunca hacia el más bajo.
"""

import json
import re
from dataclasses import dataclass

from src.groq_client import chat_completion

# --- Capa de reglas duras: señales de alarma que SIEMPRE elevan a rojo,
# sin importar lo que decida el LLM. Ajusta esta lista según los casos
# de tu dataset (trayectorias_postop_silver.xlsx te puede dar pistas de
# qué síntomas marcan los casos "rojo" reales).
SEÑALES_ALARMA_ROJO = [
    r"no puedo respirar",
    r"dificultad para respirar",
    r"sangrado abundante",
    r"sangra mucho",
    r"fiebre alta",
    r"\b39\b|\b40\b|\b41\b",  # temperaturas altas mencionadas en grados
    r"dolor de pecho",
    r"desmay",
    r"pérdida de conciencia",
    r"vómito con sangre",
    r"no puedo mover",
]

SEÑALES_ALARMA_AMARILLO = [
    r"fiebre",
    r"hinchaz[oó]n",
    r"enrojecimiento",
    r"pus",
    r"mal olor",
    r"dolor.*no mejora",
    r"dolor.*aumenta",
]


def detectar_reglas(texto_paciente: str) -> str:
    """Devuelve 'rojo', 'amarillo' o 'verde' según coincidencias de palabras clave."""
    texto_lower = texto_paciente.lower()
    for patron in SEÑALES_ALARMA_ROJO:
        if re.search(patron, texto_lower):
            return "rojo"
    for patron in SEÑALES_ALARMA_AMARILLO:
        if re.search(patron, texto_lower):
            return "amarillo"
    return "verde"


SYSTEM_INSTRUCTION_DECISION = """Eres un sistema de triaje para seguimiento postoperatorio. \
Analiza lo que reporta el paciente y clasifica la situación en una de tres categorías:

- "rojo": síntomas que requieren atención médica urgente/inmediata (dificultad respiratoria, \
sangrado abundante, fiebre alta, dolor torácico, pérdida de conciencia, signos de infección grave).
- "amarillo": síntomas que ameritan seguimiento cercano pero no son emergencia inmediata \
(fiebre leve, dolor que empeora, signos tempranos de infección, dudas que requieren evaluación pronta).
- "verde": recuperación dentro de lo esperado, sin señales de alarma.

Si la información del paciente es AMBIGUA o INSUFICIENTE para decidir con confianza, NO asumas \
"verde" por defecto — responde con "requiere_mas_info": true y formula una pregunta de seguimiento \
específica antes de clasificar.

IMPORTANTE: Distingue entre dos tipos de mensajes del paciente:

1. REPORTE DE SÍNTOMAS ("me duele", "tengo fiebre", "la herida está roja") — aquí sí \
aplica el triaje completo, y si la información es ambigua o insuficiente para evaluar \
la gravedad, pide más información.

2. PREGUNTA INFORMATIVA sobre cuidados ("¿puedo bañarme?", "¿cuándo puedo caminar?", \
"¿qué puedo comer?") — estas NO son reportes de síntomas. Clasifícalas como "verde" con \
requiere_mas_info: false, para que el sistema pueda responder la consulta con la \
información clínica disponible. Solo pide más información si la pregunta es genuinamente \
incomprensible.

Responde ÚNICAMENTE con un objeto JSON, sin texto adicional, con este formato exacto:
{
  "clasificacion": "rojo" | "amarillo" | "verde",
  "razon": "explicación breve basada en lo que dijo el paciente",
  "requiere_mas_info": true | false,
  "pregunta_seguimiento": "pregunta a hacer si requiere_mas_info es true, si no, cadena vacía"
}"""


@dataclass
class DecisionResult:
    clasificacion: str  # verde | amarillo | rojo
    razon: str
    requiere_mas_info: bool
    pregunta_seguimiento: str
    fuente_decision: str  # "llm", "reglas", o "llm+reglas" si coincidieron/escaló


def clasificar_turno(texto_paciente: str, historial: list[dict] | None = None) -> DecisionResult:
    # 1. Reglas duras (rápido, determinístico, red de seguridad)
    clasificacion_reglas = detectar_reglas(texto_paciente)

    # Contexto conversacional: permite entender respuestas breves ("sí", "no",
    # "tres días") que dependen de lo que el agente preguntó en turnos previos.
    bloque_historial = ""
    if historial:
        lineas = []
        for turno in historial:
            etiqueta = "Paciente" if turno["rol"] == "paciente" else "Agente"
            lineas.append(f"{etiqueta}: {turno['texto']}")
        bloque_historial = "\n\nHISTORIAL PREVIO DE LA CONVERSACIÓN:\n" + "\n".join(lineas)

    # 2. LLM con salida estructurada
    resultado_llm = chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION_DECISION},
            {"role": "user", "content": f"Reporte del paciente: {texto_paciente}{bloque_historial}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,  # baja temperatura: queremos consistencia, no creatividad, en triaje
    )

    try:
        datos_llm = json.loads(resultado_llm.text)
    except (json.JSONDecodeError, TypeError):
        # Si el LLM no devolvió JSON válido, no asumas "verde" — cae a lo que digan las reglas
        datos_llm = {
            "clasificacion": clasificacion_reglas,
            "razon": "El LLM no devolvió una clasificación válida; se usó la capa de reglas.",
            "requiere_mas_info": False,
            "pregunta_seguimiento": "",
        }

    # Si el LLM pidió más información, puede no traer una clasificación definitiva.
    # Nunca asumas "verde" en ese caso — usa "amarillo" como estado provisional
    # hasta que se resuelva la ambigüedad (asimetría clínica: ante la duda, no minimizar).
    if datos_llm.get("requiere_mas_info") and not datos_llm.get("clasificacion"):
        datos_llm["clasificacion"] = "amarillo"
        if not datos_llm.get("razon"):
            datos_llm["razon"] = (
                "Información insuficiente para clasificar con confianza; "
                "estado provisional en amarillo hasta indagar más."
            )

    # 3. Combina: la decisión final es la MÁS ALTA entre LLM y reglas (nunca la más baja)
    orden = {"verde": 0, "amarillo": 1, "rojo": 2}
    clasificacion_llm = datos_llm.get("clasificacion") or "verde"

    if orden.get(clasificacion_reglas, 0) > orden.get(clasificacion_llm, 0):
        clasificacion_final = clasificacion_reglas
        razon_final = (
            f"{datos_llm.get('razon', '')} "
            "[Elevado por regla de seguridad: coincidencia con señal de alarma explícita.]"
        )
        fuente = "reglas"
    else:
        clasificacion_final = clasificacion_llm
        razon_final = datos_llm.get("razon", "")
        fuente = "llm" if clasificacion_reglas == "verde" else "llm+reglas"

    return DecisionResult(
        clasificacion=clasificacion_final,
        razon=razon_final,
        requiere_mas_info=datos_llm.get("requiere_mas_info", False),
        pregunta_seguimiento=datos_llm.get("pregunta_seguimiento", ""),
        fuente_decision=fuente,
    )


if __name__ == "__main__":
    casos_prueba = [
        "Me duele un poco la herida pero es soportable.",
        "Tengo fiebre desde ayer y la herida está roja e hinchada.",
        "No puedo respirar bien y siento mucho dolor en el pecho.",
        "Estoy bien, gracias.",
        "No sé, algo se siente raro pero no sabría decir qué.",
    ]
    for caso in casos_prueba:
        r = clasificar_turno(caso)
        print(f"\n'{caso}'")
        print(f"  → {r.clasificacion.upper()} (fuente: {r.fuente_decision})")
        print(f"  Razón: {r.razon}")
        if r.requiere_mas_info:
            print(f"  Pregunta de seguimiento: {r.pregunta_seguimiento}")