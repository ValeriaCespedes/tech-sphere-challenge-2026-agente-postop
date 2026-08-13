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


def recuperar(pregunta: str, n_results: int = 4, escenario: str | None = None) -> list[dict]:
    """Busca dentro del escenario del paciente y, en paralelo, en todo el
    corpus sin filtro. Usa el conjunto de resultados con la MEJOR distancia
    real, no simplemente el primero que tenga algo — así no se pierde un
    documento muy relevante (ej. subido por consola) solo porque el
    escenario del paciente tuvo algún resultado mediocre."""

    def _buscar(where_filtro):
        kwargs = {"query_texts": [pregunta], "n_results": n_results}
        if where_filtro:
            kwargs["where"] = where_filtro
        resultados = _collection.query(**kwargs)
        if not resultados["ids"][0]:
            return []
        encontrados = []
        for doc, meta, dist in zip(
            resultados["documents"][0], resultados["metadatas"][0], resultados["distances"][0]
        ):
            if dist <= UMBRAL_DISTANCIA_MAX:
                encontrados.append({
                    "texto": doc, "archivo": meta["archivo"], "ruta": meta["ruta"],
                    "escenario": meta["escenario"], "distancia": dist,
                })
        return encontrados

    resultados_sin_filtro = _buscar(None)

    if not escenario:
        return resultados_sin_filtro

    resultados_escenario = _buscar({"escenario": escenario})

    # Si no hay resultados en ninguno de los dos, no hay nada que devolver
    if not resultados_escenario and not resultados_sin_filtro:
        return []
    if not resultados_escenario:
        return resultados_sin_filtro
    if not resultados_sin_filtro:
        return resultados_escenario

    # Compara la mejor distancia (más baja = más relevante) de cada conjunto
    mejor_escenario = min(r["distancia"] for r in resultados_escenario)
    mejor_sin_filtro = min(r["distancia"] for r in resultados_sin_filtro)

    # Si el resultado sin filtro es claramente mejor, lo utiliza (esto rescata
    # documentos de consola o de otro escenario que sean muy relevantes).
    # Si son similares, prioriza el escenario del paciente por precisión clínica.
    if mejor_sin_filtro < mejor_escenario - 0.03:
        return resultados_sin_filtro
    return resultados_escenario