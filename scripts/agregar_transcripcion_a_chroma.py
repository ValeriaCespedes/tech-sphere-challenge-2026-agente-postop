# scripts/agregar_transcripcion_a_chroma.py
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def chunkear(texto, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    inicio = 0
    while inicio < len(texto):
        chunks.append(texto[inicio:inicio + size])
        inicio += size - overlap
    return [c.strip() for c in chunks if len(c.strip()) > 30]

def main():
    texto = Path("data/appendicitis_transcripcion.txt").read_text(encoding="utf-8")
    chunks = chunkear(texto)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-small"
    )
    client = chromadb.PersistentClient(path="./data/chroma")
    collection = client.get_or_create_collection(
        name="corpus_clinico", embedding_function=ef
    )

    nombre = "Revisión de la literatura sobre apendicitis aguda pediátrica (transcripción manual).pdf"
    ids = [f"appendicitis_manual_{i}" for i in range(len(chunks))]
    metas = [{
        "archivo": nombre,
        "ruta": "Appendicitis/" + nombre,
        "escenario": "Appendicitis",
        "chunk_idx": i,
        "fuente_transcripcion_manual": True,  # trazabilidad: no vino de extracción automática
    } for i in range(len(chunks))]

    collection.add(ids=ids, documents=chunks, metadatas=metas)
    print(f"Agregados {len(chunks)} chunks (transcripción manual revisada).")

if __name__ == "__main__":
    main()