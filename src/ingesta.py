import re
import hashlib
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_TEXT_LEN = 50
CHROMA_DIR = "./data/chroma"

_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-small"
)
_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(
    name="corpus_clinico", embedding_function=_ef
)


def normalizar_texto_espaciado(texto: str) -> str:
    lineas_corregidas = []
    for linea in texto.split("\n"):
        tokens = linea.split(" ")
        tokens_no_vacios = [t for t in tokens if t]
        if tokens_no_vacios and sum(1 for t in tokens_no_vacios if len(t) == 1) / len(tokens_no_vacios) > 0.6:
            linea = re.sub(r'(?<=\S) (?=\S)', '', linea.replace("  ", "␟"))
            linea = linea.replace("␟", " ")
        lineas_corregidas.append(linea)
    return "\n".join(lineas_corregidas)


def extraer_texto_desde_bytes(pdf_bytes: bytes, nombre_archivo: str) -> str:
    """Extrae texto de un PDF recibido como bytes (subido desde la consola)."""
    import io
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        texto_crudo = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"⚠️  Error leyendo {nombre_archivo}: {e}")
        return ""

    try:
        return normalizar_texto_espaciado(texto_crudo)
    except Exception as e:
        print(f"⚠️  Error normalizando {nombre_archivo}: {e}")
        return texto_crudo


def chunkear(texto: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    chunks = []
    inicio = 0
    while inicio < len(texto):
        chunks.append(texto[inicio:inicio + size])
        inicio += size - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 30]


def hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def indexar_documento(pdf_bytes: bytes, nombre_archivo: str, escenario: str = "consola_admin") -> dict:
    """
    Indexa un documento subido desde la consola. Devuelve un resumen del resultado.
    Si el documento ya existe (mismo nombre), lo reemplaza (borra chunks previos primero).
    """
    # si ya existe un documento con ese nombre, bórralo antes de reindexar (evita duplicados)
    eliminar_documento(nombre_archivo)

    texto = extraer_texto_desde_bytes(pdf_bytes, nombre_archivo)
    if len(texto.strip()) < MIN_TEXT_LEN:
        return {"ok": False, "motivo": "Sin texto útil (probable escaneo sin OCR)", "chunks": 0}

    chunks = chunkear(texto)
    if not chunks:
        return {"ok": False, "motivo": "No se generaron chunks válidos", "chunks": 0}

    doc_hash = hash_bytes(pdf_bytes)
    ids = [f"{doc_hash}_{i}" for i in range(len(chunks))]
    metas = [{
        "archivo": nombre_archivo,
        "ruta": f"{escenario}/{nombre_archivo}",
        "escenario": escenario,
        "chunk_idx": i,
        "fuente_consola_admin": True,
    } for i in range(len(chunks))]

    _collection.add(ids=ids, documents=chunks, metadatas=metas)
    return {"ok": True, "motivo": "Procesado y disponible", "chunks": len(chunks)}


def listar_documentos() -> list[dict]:
    """Devuelve la lista de documentos únicos indexados, con su cantidad de chunks."""
    todos = _collection.get()
    conteo = {}
    for meta in todos["metadatas"]:
        archivo = meta["archivo"]
        conteo[archivo] = conteo.get(archivo, 0) + 1
    return [{"archivo": a, "chunks": c} for a, c in sorted(conteo.items())]


def eliminar_documento(nombre_archivo: str) -> int:
    """Elimina todos los chunks de un documento por nombre de archivo. Devuelve cuántos borró."""
    existentes = _collection.get(where={"archivo": nombre_archivo})
    if existentes["ids"]:
        _collection.delete(ids=existentes["ids"])
    return len(existentes["ids"])