import re
import hashlib
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

#Para saltar el límite de caracteres
import os

def ruta_segura(path: Path) -> Path:
    """En Windows, evita el límite de 260 caracteres de MAX_PATH."""
    if os.name == "nt":
        abs_path = str(path.resolve())
        if not abs_path.startswith("\\\\?\\"):
            abs_path = "\\\\?\\" + abs_path
        return Path(abs_path)
    return path







TEXTOS_DIR = Path("dataset/textos")
CHROMA_DIR = "./data/chroma"
CHUNK_SIZE = 800      # caracteres por chunk
CHUNK_OVERLAP = 150
MIN_TEXT_LEN = 50     # umbral para detectar PDFs sin capa de texto


def extraer_texto(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(ruta_segura(pdf_path)))
        texto = "\n".join(page.extract_text() or "" for page in reader.pages)
        return normalizar_texto_espaciado(texto)
    except Exception as e:
        print(f" Error leyendo {pdf_path.name}: {e}")
        return ""


def hash_archivo(path: Path) -> str:
    return hashlib.md5(ruta_segura(path).read_bytes()).hexdigest()

def chunkear(texto: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    chunks = []
    inicio = 0
    while inicio < len(texto):
        chunks.append(texto[inicio:inicio + size])
        inicio += size - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 30]


def main():
    pdfs = list(TEXTOS_DIR.rglob("*.pdf"))
    print(f"Encontrados {len(pdfs)} PDFs en {TEXTOS_DIR}")

    # dedupe por contenido, no por nombre
    vistos = {}
    unicos = []
    for p in pdfs:
        h = hash_archivo(p)
        if h in vistos:
            print(f"  ↳ duplicado: {p.relative_to(TEXTOS_DIR)} == {vistos[h].relative_to(TEXTOS_DIR)}")
            continue
        vistos[h] = p
        unicos.append(p)
    print(f"{len(unicos)} PDFs únicos tras deduplicar")

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-small"
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name="corpus_clinico", embedding_function=ef
    )

    excluidos = []
    ids, docs, metas = [], [], []

    for i, pdf_path in enumerate(unicos, 1):
        print(f"[{i}/{len(unicos)}] Procesando: {pdf_path.name[:60]}...")
        texto = extraer_texto(pdf_path)
        if len(texto.strip()) < MIN_TEXT_LEN:
            print(f"⚠️  Sin texto útil (probable escaneo): {pdf_path.relative_to(TEXTOS_DIR)}")
            excluidos.append(str(pdf_path.relative_to(TEXTOS_DIR)))
            continue  # ver nota de OCR abajo

        escenario = pdf_path.parent.name  # carpeta = escenario
        chunks = chunkear(texto)
        for i, chunk in enumerate(chunks):
            ids.append(f"{pdf_path.stem}_{i}")
            docs.append(chunk)
            metas.append({
                "archivo": pdf_path.name,
                "ruta": str(pdf_path.relative_to(TEXTOS_DIR)),
                "escenario": escenario,
                "chunk_idx": i,
            })

    # Chroma acepta añadir en lotes; con corpus grande, hazlo por bloques de 500
    for i in range(0, len(ids), 500):
        collection.add(
            ids=ids[i:i+500],
            documents=docs[i:i+500],
            metadatas=metas[i:i+500],
        )

    print(f"\nIndexados {len(ids)} chunks de {len(unicos) - len(excluidos)} documentos.")
    if excluidos:
        print(f"Excluidos ({len(excluidos)}) por no tener capa de texto:")
        for e in excluidos:
            print(f"  - {e}")
        # deja constancia en disco para citarlo en el informe final
        Path("data").mkdir(exist_ok=True)
        Path("data/excluidos_rag.json").write_text(
            json.dumps(excluidos, indent=2, ensure_ascii=False)
        )
##Añadí esta parte para evitar los problemas de espaciado
def normalizar_texto_espaciado(texto: str) -> str:
    """
    Corrige el patrón 'P L A N  D E' -> 'PLAN DE', típico de PDFs de diseño
    gráfico donde el texto quedó con letra suelta y espacio.
    Detecta líneas con muchas letras sueltas separadas por espacio simple
    y colapsa esos espacios.
    """
    lineas_corregidas = []
    for linea in texto.split("\n"):
        # heurística: si más del 60% de los "tokens" de la línea son de 1 char,
        # es candidata a texto espaciado letra por letra
        tokens = linea.split(" ")
        tokens_no_vacios = [t for t in tokens if t]
        if tokens_no_vacios and sum(1 for t in tokens_no_vacios if len(t) == 1) / len(tokens_no_vacios) > 0.6:
            # colapsa espacios simples entre letras, conserva dobles espacios como separador de palabra real
            linea = re.sub(r'(?<=\S) (?=\S)', '', linea.replace("  ", "␟"))
            linea = linea.replace("␟", " ")
        lineas_corregidas.append(linea)
    return "\n".join(lineas_corregidas)

if __name__ == "__main__":
    main()




