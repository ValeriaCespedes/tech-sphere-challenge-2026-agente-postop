from src.rag import _collection
r = _collection.query(query_texts=["cuidados en casa después de la cirugía de apendicitis"], n_results=10)
for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
    print(f"{dist:.3f} | {meta['archivo']}")