from src.rag import _collection

resultados = _collection.query(
    query_texts=["dolor tres días después de apendicectomía"],
    n_results=10,
)

for doc, meta, dist in zip(
    resultados["documents"][0],
    resultados["metadatas"][0],
    resultados["distances"][0],
):
    print(f"{dist:.3f} | {meta['archivo']}")
    print(f"   {doc[:120]}...")
    print()