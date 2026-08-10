# Cómo integrar este esqueleto en tu repo del reto

Este esqueleto se generó por fuera de tu repositorio real (no tengo acceso a
tu clon local ni al dataset todavía). Pasos para integrarlo:

1. Copia `requirements.txt`, `.env.example`, `src/`, `scripts/` a la raíz de
   tu repo clonado del reto (el que ya trae `dataset/`, `docs/` y `LICENSE`).
2. `python -m venv .venv && source .venv/bin/activate` (o el equivalente en
   tu SO) y `pip install -r requirements.txt`.
3. `cp .env.example .env` y completa tu `GROQ_API_KEY`.
4. Prueba la conexión a Groq:
   ```
   python -m src.groq_client
   ```
   Deberías ver una línea con la latencia y una respuesta corta del modelo.
5. Corre la exploración del dataset (ajusta `--dataset-dir` si tu carpeta
   `dataset/` no está en la raíz):
   ```
   python scripts/explore_dataset.py --dataset-dir ./dataset
   ```
   **Revisa la sección "1) INSPECCIÓN DE COLUMNAS" del output.** Los nombres
   de columna en el script están inferidos de la descripción del README, no
   verificados contra el archivo real. Si algo no calza, ajusta las
   constantes `COL_*` al inicio de `scripts/explore_dataset.py`.
6. Corre el inventario del corpus:
   ```
   python scripts/inventory_corpus.py --textos-dir ./dataset/textos
   ```

## Qué falta después de esto (siguiente bloque del Día 1 tarde)

- Pipeline de chunking + embeddings (BGE-M3) + indexado en ChromaDB.
- Decidir qué hacer con el/los PDF(s) sin capa de texto que reporte el
  inventario (OCR o exclusión justificada).

## Nota sobre el modelo de Groq

El identificador exacto del modelo (`GROQ_LLM_MODEL` en `.env`) puede haber
cambiado desde que se escribió este esqueleto. Verifica el ID vigente de
Llama 3.1 70B en https://console.groq.com/docs/models antes de tu primera
corrida real, y corrígelo en tu `.env` si es necesario.
