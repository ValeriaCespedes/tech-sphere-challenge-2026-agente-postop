import os
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = os.environ.get("CHROMA_DB_DIR", "./data/chroma")
UMBRAL_DISTANCIA_MAX = 0.45

_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-small"
)
_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(
    name="corpus_clinico", embedding_function=_ef
)


def recuperar(pregunta: str, n_results: int = 4) -> list[dict]:
    """Devuelve chunks relevantes con su fuente, o lista vacía si nada supera el umbral."""
    resultados = _collection.query(query_texts=[pregunta], n_results=n_results)

    if not resultados["ids"][0]:
        return []

    encontrados = []
    for doc, meta, dist in zip(
        resultados["documents"][0],
        resultados["metadatas"][0],
        resultados["distances"][0],
    ):
        if dist <= UMBRAL_DISTANCIA_MAX:
            encontrados.append({
                "texto": doc,
                "archivo": meta["archivo"],
                "ruta": meta["ruta"],
                "escenario": meta["escenario"],
                "distancia": dist,
            })
    return encontrados
