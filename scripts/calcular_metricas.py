"""
Calcula las métricas obligatorias del README a partir de los registros
de llamadas ya persistidos en data/llamadas/*.json:
  - Latencia P50 y P95 (desde que el paciente termina de hablar hasta
    que empieza a sonar el audio del agente)
  - Consumo de tokens (entrada/salida por turno y por llamada)
  - Invocaciones al modelo por turno
  - Consultas al RAG por llamada
  - Costo estimado por llamada, extrapolado a precios de API de producción
"""

import json
import statistics
from pathlib import Path

DIR_LLAMADAS = Path("data/llamadas")

# Precios de referencia de Groq para Llama 3.3 70B Versatile (USD por millón
# de tokens). Verifica el valor vigente en https://groq.com/pricing/ antes
# de reportarlo como definitivo en el README — los precios cambian.
PRECIO_INPUT_POR_MILLON = 0.59
PRECIO_OUTPUT_POR_MILLON = 0.79


def cargar_todas_las_llamadas():
    llamadas = []
    for archivo in DIR_LLAMADAS.glob("*.json"):
        try:
            datos = json.loads(archivo.read_text(encoding="utf-8"))
            if datos.get("turnos"):  # descarta llamadas vacías (sin turnos)
                llamadas.append(datos)
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"⚠️  No se pudo leer {archivo.name}, se omite.")
    return llamadas


def main():
    llamadas = cargar_todas_las_llamadas()
    print(f"Llamadas con al menos un turno: {len(llamadas)}")

    todos_los_turnos = []
    for llamada in llamadas:
        todos_los_turnos.extend(llamada["turnos"])

    print(f"Total de turnos analizados: {len(todos_los_turnos)}\n")

    if not todos_los_turnos:
        print("No hay turnos para analizar. Genera algunas llamadas de prueba primero.")
        return

    # --- Latencia ---
    latencias = [t["latencia_total"] for t in todos_los_turnos if t.get("latencia_total") is not None]
    latencias_ordenadas = sorted(latencias)

    def percentil(datos, p):
        if not datos:
            return None
        k = (len(datos) - 1) * (p / 100)
        f = int(k)
        c = min(f + 1, len(datos) - 1)
        if f == c:
            return datos[f]
        return datos[f] + (datos[c] - datos[f]) * (k - f)

    p50 = percentil(latencias_ordenadas, 50)
    p95 = percentil(latencias_ordenadas, 95)

    print("=== LATENCIA (segundos, desde que termina de hablar el paciente hasta que suena el audio) ===")
    print(f"P50: {p50:.2f}s" if p50 else "P50: sin datos")
    print(f"P95: {p95:.2f}s" if p95 else "P95: sin datos")
    print(f"Mínima: {min(latencias):.2f}s | Máxima: {max(latencias):.2f}s")
    print()

    # --- Desglose de latencia por etapa (promedio) ---
    stt = [t["latencia_stt"] for t in todos_los_turnos if t.get("latencia_stt") is not None]
    llm = [t["latencia_llm"] for t in todos_los_turnos if t.get("latencia_llm") is not None]
    tts = [t["latencia_tts"] for t in todos_los_turnos if t.get("latencia_tts") is not None]

    print("=== DESGLOSE PROMEDIO POR ETAPA ===")
    if stt:
        print(f"STT (transcripción): {statistics.mean(stt):.2f}s promedio")
    if llm:
        print(f"LLM (RAG + razonamiento): {statistics.mean(llm):.2f}s promedio")
    if tts:
        print(f"TTS (síntesis de voz): {statistics.mean(tts):.2f}s promedio")
    print()

    # --- Tokens ---
    tokens_input = [t["tokens"]["input"] for t in todos_los_turnos if t.get("tokens") and t["tokens"].get("input") is not None]
    tokens_output = [t["tokens"]["output"] for t in todos_los_turnos if t.get("tokens") and t["tokens"].get("output") is not None]

    print("=== CONSUMO DE TOKENS ===")
    if tokens_input:
        print(f"Tokens de entrada por turno — promedio: {statistics.mean(tokens_input):.0f} | total: {sum(tokens_input)}")
    if tokens_output:
        print(f"Tokens de salida por turno — promedio: {statistics.mean(tokens_output):.0f} | total: {sum(tokens_output)}")
    print()

    # --- Invocaciones al modelo por turno ---
    # Cada turno hace 2 invocaciones al LLM: una para clasificar (decision.py)
    # y otra para responder (responder.py). Esto es fijo por diseño actual.
    print("=== INVOCACIONES AL MODELO ===")
    print("Invocaciones al LLM por turno: 2 (clasificación de triaje + generación de respuesta)")
    print()

    # --- Consultas al RAG por llamada ---
    consultas_rag_por_llamada = [len(llamada["turnos"]) for llamada in llamadas]
    print("=== CONSULTAS AL RAG ===")
    print(f"Consultas al RAG por llamada — promedio: {statistics.mean(consultas_rag_por_llamada):.1f}")
    print("(1 consulta de retrieval por turno, dentro de responder_con_contexto)")
    print()

    # --- Costo estimado por llamada ---
    if tokens_input and tokens_output:
        promedio_input_por_turno = statistics.mean(tokens_input)
        promedio_output_por_turno = statistics.mean(tokens_output)
        promedio_turnos_por_llamada = statistics.mean(consultas_rag_por_llamada)

        costo_input_por_turno = (promedio_input_por_turno / 1_000_000) * PRECIO_INPUT_POR_MILLON
        costo_output_por_turno = (promedio_output_por_turno / 1_000_000) * PRECIO_OUTPUT_POR_MILLON
        costo_por_turno = costo_input_por_turno + costo_output_por_turno
        costo_por_llamada = costo_por_turno * promedio_turnos_por_llamada

        print("=== COSTO ESTIMADO POR LLAMADA ===")
        print(f"(Nota: esto NO incluye el costo de clasificación de triaje, que es una")
        print(f" segunda llamada al LLM por turno — multiplicar aprox. x2 para estimación conservadora)")
        print(f"Turnos promedio por llamada: {promedio_turnos_por_llamada:.1f}")
        print(f"Costo por turno (solo respuesta RAG): ${costo_por_turno:.5f} USD")
        print(f"Costo estimado por llamada (solo respuesta RAG): ${costo_por_llamada:.5f} USD")
        print(f"Costo estimado por llamada (incluyendo triaje, x2 aprox.): ${costo_por_llamada * 2:.5f} USD")
        print()
        print("Precios usados: Groq Llama 3.3 70B Versatile —")
        print(f"  ${PRECIO_INPUT_POR_MILLON}/M tokens entrada, ${PRECIO_OUTPUT_POR_MILLON}/M tokens salida")
        print("  (verificar precio vigente en https://groq.com/pricing/ antes de publicar)")


if __name__ == "__main__":
    main()