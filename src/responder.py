from src.rag import recuperar
from src.groq_client import chat_completion

SYSTEM_INSTRUCTION = """Eres un agente de seguimiento postoperatorio en español, hablando con \
pacientes colombianos. SOLO puedes dar información clínica que esté EXPLÍCITAMENTE respaldada \
por el CONTEXTO que se te entrega en cada turno — no infieras, no generalices ni sintetices \
más allá de lo que el texto dice literalmente. Si el CONTEXTO no contiene una respuesta directa \
a la pregunta del paciente —aunque contenga información relacionada al tema— dilo explícitamente \
y ofrece escalar a personal médico, en vez de construir una respuesta razonable a partir de \
fragmentos parciales. Nunca inventes una dosis, medicamento, procedimiento, ni tranquilices \
sobre un síntoma de alarma sin respaldo textual directo del contexto."""


def responder_con_contexto(pregunta_paciente: str) -> dict:
    fuentes = recuperar(pregunta_paciente)

    if not fuentes:
        return {
            "respuesta": (
                "No tengo información suficiente en mis protocolos para responder eso "
                "con seguridad. Voy a poner esto en conocimiento del equipo médico."
            ),
            "fuentes": [],
            "sin_conocimiento": True,
            "latencia": None,
            "tokens": None,
        }

    contexto = "\n\n".join(
        f"[Fuente: {f['archivo']}]\n{f['texto']}" for f in fuentes
    )

    resultado = chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"""CONTEXTO:
{contexto}

PREGUNTA DEL PACIENTE:
{pregunta_paciente}

Responde en español, en tono cálido y profesional, en máximo 3 frases (esto es una \
llamada de voz, no un chat). Basa tu respuesta únicamente en el CONTEXTO."""},
        ]
    )

    return {
        "respuesta": resultado.text,
        "fuentes": list({f["archivo"]: {"archivo": f["archivo"], "ruta": f["ruta"]} for f in fuentes}.values()),
        "sin_conocimiento": False,
        "latencia": resultado.latency_seconds,
        "tokens": {"input": resultado.input_tokens, "output": resultado.output_tokens},
    }


if __name__ == "__main__":
    r = responder_con_contexto("¿Es normal tener dolor tres días después de una apendicectomía?")
    #r = responder_con_contexto("¿En qué consiste un circuito RC de primer orden?")
    print(r["respuesta"])
    print("Fuentes:", r["fuentes"])
    print(f"Latencia: {r['latencia']:.2f}s | Tokens: {r['tokens']}")