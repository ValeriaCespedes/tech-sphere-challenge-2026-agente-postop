from src.rag import _collection

r = _collection.get(
    where={"archivo": "PLAN DE CUIDADO EN CASA DE PACIENTE EN POSTOPERATORIO DE APENDICECTOMÍA.pdf"}
)
print("Chunks encontrados:", len(r["ids"]))
for i, doc in enumerate(r["documents"]):
    print(f"\n--- Chunk {i} ---")
    print(doc[:300])