# Tech Sphere Challenge 2026 — Repositorio base

**Vas a construir un agente de voz con IA para seguimiento postoperatorio.**

Un paciente sale de un procedimiento y necesita que alguien esté pendiente de él en las
primeras horas. Tu agente hace esa llamada: conversa con el paciente, entiende sus
síntomas con información clínica real, y decide cuándo alertar a personal capacitado.

Este es el **repositorio base del reto**. Clónalo: aquí están los datos con los que vas
a trabajar, la definición de lo que se espera de tu solución y las reglas con las que se
va a evaluar.

- **Cómo se evalúa tu entrega** → [`docs/rubrica-evaluacion.md`](docs/rubrica-evaluacion.md)
- **Stack abierto y modelos permitidos** → [`docs/stack-tecnico.md`](docs/stack-tecnico.md)
- **Los datos** → [`dataset/`](dataset/)

---

## El problema

El seguimiento postoperatorio depende hoy de personal humano: es costoso, no escala y
está sujeto a errores. El paciente, mientras tanto, no tiene conocimiento médico —a veces
ni un termómetro— y describe lo que siente en lenguaje cotidiano, ambiguo y regional:

> *"Me duele como aquí abajito de la axila hace como 20 minutos."*

En paralelo, la operación clínica vive en conocimiento no estructurado —manuales,
instructivos, guías, PDFs, notas— que **cambia de versión constantemente**. El agente
debe reflejar siempre la versión vigente sin contaminarse con la anterior.

Tres cosas hacen este reto distinto de un chatbot cualquiera:

- **Es voz, no chat.** Conversación en tiempo real, con todo lo que eso implica:
  latencia, silencios incómodos, respuestas largas inviables.
- **Es salud, no e-commerce.** Cero tolerancia a alucinaciones, respuestas fundamentadas
  en el corpus clínico, y honestidad explícita cuando el agente no sabe.
- **El conocimiento es vivo, no estático.** El RAG debe poder actualizarse —aprender y
  olvidar— en caliente.

## Qué construyes

- Una conversación de voz que se adapta a las respuestas del paciente.
- Respuestas fundamentadas en una base de conocimiento clínico (RAG).
- Una consola para actualizar el conocimiento en caliente: subes un documento y el agente
  lo aprende; lo eliminas y lo olvida.
- Trazabilidad: cada respuesta clínica registra qué documento la sustenta.
- Una lógica de decisión: ¿esto amerita alertar a un humano, o no?
- Un resumen estructurado de cada llamada.

### Qué no necesitas construir

Telefonía real en producción · integración con sistemas hospitalarios reales ·
autenticación empresarial o gestión de roles · cobertura de todos los procedimientos
médicos existentes.

## Mi solución

Agente de voz para seguimiento postoperatorio, construido con FastAPI, RAG sobre
ChromaDB, y Groq (Llama 3.3 70B para razonamiento, Whisper Large V3 para
transcripción) + Kokoro TTS para síntesis de voz en español.

### Las dos superficies

Tu solución debe exponer dos superficies. Pueden ser una sola aplicación o dos; el diseño
visual no se evalúa, pero el contrato funcional sí:

| Superficie | Qué representa | Contrato funcional mínimo |
|---|---|---|
| **Consola de administración** | El back-office del producto real: gestión del conocimiento | Subir documento · listar documentos cargados · eliminar documento · indicación visible de "procesado y disponible" |
| **Interfaz de llamada** | La llamada telefónica de producción | Iniciar llamada de voz desde el navegador · hablar (micrófono) · escuchar al agente |

Puedes ofrecer además API, CLI o una carpeta que el sistema vigile e ingiera
automáticamente, pero la consola es exigida.

### Modelo de lenguaje usado (G3)

**Llama 3.3 70B Versatile, vía Groq.** Se eligió por su latencia ultra-baja
(clave para una conversación de voz fluida) y porque, según la nota de
`stack-tecnico.md` ("los modelos vencen, las familias no"), es el sucesor
vigente de la familia Llama 3.1 en Groq — el modelo originalmente listado
(`llama-3.1-70b-versatile`) fue descontinuado por el proveedor durante la
ventana del reto. Se evaluó también Google Gemini 1.5 Flash como alternativa
(código conservado en `src/gemini_client.py`), pero se descartó por mayor
latencia y por presentar también discontinuidad del modelo listado en la
plataforma de Google durante el desarrollo.

### Instalación

**Requisitos previos:**
- Python 3.11
- Cuenta y API key de Groq (https://console.groq.com/)
- **Windows:** habilitar rutas largas antes de instalar dependencias
  (PyTorch incluye rutas de archivo muy anidadas):
  1. Ejecutar `regedit` como administrador
  2. Ir a `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
  3. Crear/editar el DWORD `LongPathsEnabled` = `1`
  4. Reiniciar el equipo
- **espeak-ng** instalado a nivel de sistema (requerido por Kokoro TTS):
  - Descargar desde https://github.com/espeak-ng/espeak-ng/releases
  - Instalar y agregar la carpeta de instalación al PATH del sistema
  - Verificar con `espeak-ng --version`

**Pasos:**

```bash
git clone https://github.com/ValeriaCespedes/tech-sphere-challenge-2026-agente-postop.git
cd tech-sphere-challenge-2026-agente-postop

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # Linux/Mac

pip install -r requirements.txt

copy .env.example .env            # Windows
# cp .env.example .env            # Linux/Mac
# Editar .env y completar GROQ_API_KEY con tu propia key de console.groq.com

# El corpus clínico ya viene pre-indexado en data/chroma/ — no es necesario
# volver a correr la indexación.

uvicorn server:app --reload --port 8000
```

Abrir `http://localhost:8000/` , ahí está la interfaz de llamada, con un
selector de 5 pacientes de ejemplo (uno por cada escenario clínico del
dataset: apendicitis, colecistitis, cáncer colorrectal, reemplazo de
cadera/rodilla, mastectomía).

Para la consola de administración de conocimiento (subir/listar/eliminar
documentos del RAG), en una terminal aparte:
```bash
streamlit run app_admin.py
```
Se abre en `http://localhost:8501/`.

### Estructura del repositorio
server.py # Servidor FastAPI: endpoints de llamada, turno y resumen
app_admin.py # Consola de administración (Streamlit) — sube/lista/elimina documentos
src/
groq_client.py # Wrapper de Groq: chat_completion() y transcribe_audio()
gemini_client.py # Cliente de Gemini (alternativa evaluada, no usada en producción)
rag.py # Recuperación semántica con ChromaDB + filtro por escenario clínico
ingesta.py # Lógica de indexación reutilizada por la consola de administración
responder.py # RAG + LLM: genera respuestas con trazabilidad de fuentes
decision.py # Lógica de triaje verde/amarillo/rojo (LLM + reglas duras)
registro_llamada.py # Persistencia de turnos y llamadas en data/llamadas/
resumen.py # Genera el resumen estructurado al cierre de la llamada
tts.py # Wrapper de Kokoro TTS
scripts/
ingest_rag.py # Indexación inicial del corpus (dataset/textos/)
static/index.html # Interfaz de llamada (HTML + JS, MediaRecorder)
data/chroma/ # Base vectorial ya indexada (se incluye en el repo)
data/llamadas/ # Registros de llamadas de prueba (JSON)

### Limitaciones conocidas

- **Sin memoria entre llamadas distintas**: el historial conversacional se
  mantiene dentro de una misma llamada (`llamada_id`), pero no persiste
  entre sesiones separadas del mismo paciente.
- **Retrieval por similitud pura**: el filtro por escenario clínico mitiga
  el ruido de documentos ajenos, pero no hay re-ranking; documentos cortos
  y específicos (ejemplo. planes de cuidado de 1 página) pueden perder frente a
  papers largos en casos límite.
- **Latencia de TTS**: Kokoro en CPU es el cuello de botella principal del
  pipeline (~10s promedio de los ~11.5s de latencia total P50). Con más
  tiempo, se evaluaría Piper como alternativa más rápida.

### Métricas de rendimiento

Calculadas sobre 17 llamadas de prueba (28 turnos totales) durante el desarrollo.

**Latencia de respuesta** (desde que el paciente termina de hablar hasta que
empieza a sonar el audio del agente):
- P50: 11.56s
- P95: 22.58s
- Rango observado: 4.22s – 24.31s

Desglose promedio por etapa: STT 0.75s · LLM (RAG + razonamiento) 0.65s ·
TTS 10.41s.

**Consumo** (promedio por turno):
- Tokens de entrada: 1,295
- Tokens de salida: 89
- Invocaciones al LLM por turno: 2 (clasificación de triaje + generación de respuesta)
- Consultas al RAG por llamada: 1.6 en promedio

**Costo estimado por llamada:** $0.00275 USD, usando precios de Groq para
Llama 3.3 70B Versatile ($0.59 / $0.79 USD por millón de tokens de entrada
/ salida, verificado en groq.com/pricing).

### Restricciones

- **El stack es abierto; el modelo, no.** Orquestación, voz, RAG y embeddings los eliges
  tú, pero el modelo de lenguaje debe ser uno de los
  [permitidos](docs/stack-tecnico.md#1-los-modelos-permitidos) — y tienes que declarar en
  tu informe cuál usaste y por qué. Mismas opciones sobre la mesa: gana la ingeniería, no
  la billetera.
- La llamada va vía **navegador/API**. No hay telefonía real.
- El agente conversa en **español**, con pacientes colombianos que usan regionalismos y
  descripciones ambiguas.
- Tu repositorio debe ser **público en GitHub**, con README y dependencias declaradas.

---

## Los datos: `dataset/`

Todos los datos del reto están en la carpeta [`dataset/`](dataset/) de este repositorio.
No hay que conectarse a nada externo para obtenerlos.

Son **datos sintéticos**. Ningún paciente, nombre, cédula, dirección o EPS corresponde a
una persona real.

| Archivo | Qué es |
|---|---|
| `dataset_final.xlsx` | **Las conversaciones.** 3.991 filas × 13 columnas: una fila es un turno, no una conversación. 40 pacientes, 160 casos (uno por paciente y día postoperatorio: 1, 3, 7 y 14), dos capas de dificultad. Incluye `label_ground_truth` con la criticidad de referencia del caso —`verde`, `amarillo` o `rojo`—, constante dentro de cada `caso_id`. |
| `trayectorias_postop_silver.xlsx` | **El cuadro clínico real de cada llamada**: dolor, fiebre, movilidad, estado de la herida, apetito y sueño, más el arquetipo de recuperación. 160 filas, una por caso. Es lo que el paciente está viviendo y el agente solo puede averiguar conversando. |
| `perfiles_clinicos_pacientes_silver_contest.xlsx` | **Perfil clínico** por paciente: procedimiento, fecha de cirugía, edad, género, comorbilidades. 40 filas. |
| `perfiles_pacientes_co.xlsx` | **Demografía colombiana** sintética: nombre, dirección, ciudad, departamento, documento y EPS. 40 filas. Se derivó de una población simulada estadounidense y se adaptó a Colombia; `adaptation_fields` lista qué campos se sustituyeron. |
| `textos/` | **El corpus clínico**: 107 documentos PDF en español e inglés —guías de práctica clínica, protocolos de recuperación, papers de complicaciones postoperatorias, planes de cuidado e instructivos para el paciente—, repartidos en cinco carpetas por escenario. Es el combustible de tu RAG. |







## Qué debes entregar

| # | Entregable |
|---|---|
| **01** | **Repositorio** público en GitHub, con tu implementación completa y documentación clara |
| **02** | **Diagrama** de la arquitectura de tu solución y del flujo de decisión del agente |
| **03** | **Informe final** con evidencia de tu proceso —prompts, configuraciones, capturas del demo— y la declaración explícita de qué modelo usaste y por qué lo elegiste |
| **04** | **Video**: demo funcional con grabación de pantalla, más las [dos preguntas de cierre](docs/rubrica-evaluacion.md#las-dos-preguntas-de-cierre-del-video) respondidas frente a cámara |





