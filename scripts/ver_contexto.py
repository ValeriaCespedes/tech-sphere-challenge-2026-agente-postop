from src.rag import recuperar

fuentes = recuperar(
    "¿Es normal tener dolor tres días después de una apendicectomía?",
    n_results=8,
)
for f in fuentes:
    print(f"[{f['archivo']}] dist={f['distancia']:.3f}")
    print(f["texto"][:200])
    print("---")