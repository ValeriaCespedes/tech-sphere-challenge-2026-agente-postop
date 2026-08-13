"""
Genera el resumen estructurado de una llamada, a partir de todos los
turnos ya persistidos. No inventa datos que no estén en el registro:
si algo no se capturó durante la llamada, el resumen lo declara
explícitamente como no disponible.
"""

from src.registro_llamada import obtener_llamada

ORDEN_SEVERIDAD = {"verde": 0, "amarillo": 1, "rojo": 2}


def generar_resumen(llamada_id: str) -> dict:
    registro = obtener_llamada(llamada_id)
    turnos = registro.get("turnos", [])

    if not turnos:
        return {
            "llamada_id": llamada_id,
            "nombre_paciente": registro.get("nombre_paciente", "No identificado"),
            "procedimiento": registro.get("procedimiento", "No especificado"),
            "sintomas_reportados": [],
            "clasificacion_final": "sin_datos",
            "referencias_usadas": [],
            "proximos_pasos": "La llamada no tuvo turnos registrados.",
        }

    # Síntomas reportados: transcripciones de cada turno del paciente
    sintomas_reportados = [t["transcripcion_paciente"] for t in turnos if t.get("transcripcion_paciente")]

    # Clasificación final: la MÁS ALTA severidad detectada en cualquier turno
    # (asimetría clínica: si en algún momento de la llamada hubo un rojo,
    # la llamada se considera rojo, aunque el paciente haya dicho después
    # que se sentía mejor)
    clasificacion_final = "verde"
    turno_criterio = None
    for t in turnos:
        clasif = t.get("clasificacion", "verde")
        if ORDEN_SEVERIDAD.get(clasif, 0) > ORDEN_SEVERIDAD.get(clasificacion_final, 0):
            clasificacion_final = clasif
            turno_criterio = t

    # Referencias usadas: documentos únicos citados en cualquier turno
    referencias = {}
    for t in turnos:
        for fuente in t.get("fuentes_rag", []):
            referencias[fuente["archivo"]] = fuente.get("ruta", "")
    referencias_usadas = [{"archivo": a, "ruta": r} for a, r in referencias.items()]

    # Próximos pasos: según la clasificación final
    if clasificacion_final == "rojo":
        proximos_pasos = (
            "Se escaló a atención médica inmediata. "
            f"Motivo: {turno_criterio.get('razon_clasificacion', '')}"
        )
    elif clasificacion_final == "amarillo":
        proximos_pasos = (
            "Se registró para seguimiento cercano por el equipo médico. "
            f"Motivo: {turno_criterio.get('razon_clasificacion', '')}"
        )
    else:
        proximos_pasos = "Recuperación dentro de lo esperado. Continuar con las indicaciones postoperatorias."

    return {
        "llamada_id": llamada_id,
        "nombre_paciente": registro.get("nombre_paciente", "No identificado"),
        "procedimiento": registro.get("procedimiento", "No especificado"),
        "inicio": registro.get("inicio"),
        "sintomas_reportados": sintomas_reportados,
        "clasificacion_final": clasificacion_final,
        "referencias_usadas": referencias_usadas,
        "proximos_pasos": proximos_pasos,
        "total_turnos": len(turnos),
    }
