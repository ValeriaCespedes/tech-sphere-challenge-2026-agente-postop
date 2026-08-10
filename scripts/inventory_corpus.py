"""
Inventario de dataset/textos/ — detecta las trampas documentadas en el README:
  - Carpetas con espacios en el nombre.
  - Documentos repetidos (por nombre y por hash de contenido).
  - El PDF de Appendicitis/ escaneado sin capa de texto (y cualquier otro
    que tenga el mismo problema, sin asumir que es el único).

No decide automáticamente qué hacer con el PDF sin texto — solo lo señala.
La decisión (OCR vs. excluir con justificación en el informe) es tuya.

Uso:
    python scripts/inventory_corpus.py --textos-dir ./dataset/textos
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import pdfplumber

MIN_CHARS_PARA_CONSIDERAR_CON_TEXTO = 20  # umbral bajo, solo para descartar PDFs vacíos/escaneados


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def has_extractable_text(path: Path) -> tuple[bool, int]:
    """Devuelve (tiene_texto, num_caracteres_extraidos)."""
    total_chars = 0
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                total_chars += len(text.strip())
    except Exception as e:
        print(f"  [ERROR] No pude abrir {path.name}: {e}")
        return False, 0
    return total_chars >= MIN_CHARS_PARA_CONSIDERAR_CON_TEXTO, total_chars


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textos-dir", type=Path, default=Path("./dataset/textos"))
    args = parser.parse_args()

    root = args.textos_dir
    if not root.exists():
        print(f"No encuentro {root}. Verifica --textos-dir.")
        return 1

    print("=" * 70)
    print("1) CARPETAS CON ESPACIOS EN EL NOMBRE")
    print("=" * 70)
    carpetas_con_espacio = [p for p in root.rglob("*") if p.is_dir() and " " in p.name]
    if carpetas_con_espacio:
        for p in carpetas_con_espacio:
            print(f"  {p}")
    else:
        print("  Ninguna encontrada (o ya las renombraste).")

    print("\n" + "=" * 70)
    print("2) PDFs por carpeta (escenario)")
    print("=" * 70)
    pdfs = sorted(root.rglob("*.pdf"))
    por_carpeta = defaultdict(list)
    for p in pdfs:
        por_carpeta[p.parent.relative_to(root)].append(p)
    for carpeta, archivos in por_carpeta.items():
        print(f"  {carpeta}: {len(archivos)} PDFs")
    print(f"  Total: {len(pdfs)} PDFs")

    print("\n" + "=" * 70)
    print("3) DOCUMENTOS DUPLICADOS (mismo contenido, por hash)")
    print("=" * 70)
    hashes = defaultdict(list)
    for p in pdfs:
        hashes[hash_file(p)].append(p)
    duplicados = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    if duplicados:
        for h, paths in duplicados.items():
            print(f"  Duplicado ({len(paths)} copias):")
            for p in paths:
                print(f"    - {p.relative_to(root)}")
    else:
        print("  No se encontraron duplicados exactos por contenido.")
        print("  (Si el README dice que sí hay, puede que difieran en metadata")
        print("   pero no en contenido de texto extraíble — revisa por nombre también.)")

    print("\n" + "=" * 70)
    print("4) PDFs SIN CAPA DE TEXTO (candidatos a necesitar OCR)")
    print("=" * 70)
    sin_texto = []
    for p in pdfs:
        tiene_texto, n_chars = has_extractable_text(p)
        if not tiene_texto:
            sin_texto.append((p, n_chars))
    if sin_texto:
        for p, n_chars in sin_texto:
            print(f"  {p.relative_to(root)}  ({n_chars} caracteres extraídos)")
        print("\n  Decide para cada uno: OCR (ej. pytesseract) o excluir con")
        print("  justificación explícita en el informe final.")
    else:
        print("  Ninguno detectado con el umbral actual — si el README menciona uno")
        print("  en Appendicitis/, revisa MIN_CHARS_PARA_CONSIDERAR_CON_TEXTO o el archivo puntual.")

    return 0


if __name__ == "__main__":
    exit(main())
