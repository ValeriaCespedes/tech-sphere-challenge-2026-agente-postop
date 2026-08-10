"""
Exploración y validación del dataset del reto.

Cubre las trampas documentadas en el README:
  - Las 4 hojas .xlsx se llaman "result".
  - `comorbilidades` y `adaptation_fields` son listas JSON dentro de una celda de texto.
  - El join no es directo: caso_id = "caso_" + trayectoria_id
  - Hay dos capas de conversación (capa1_limpia / capa2_ruidosa) bajo el mismo caso_id.
  - Clases desbalanceadas: 123 verde / 25 amarillo / 12 rojo (de 160 casos).

IMPORTANTE: los nombres exactos de columna se infieren de la descripción del
README, pero no se han verificado contra los archivos reales (aún no los
tengo disponibles). La primera vez que corras esto, revisa el bloque
"1) INSPECCIÓN DE COLUMNAS" — si algún nombre no calza, ajústalo ahí antes
de seguir. El script está escrito para fallar con un mensaje claro en vez
de romperse en silencio.

Uso:
    python scripts/explore_dataset.py --dataset-dir ./dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SHEET_NAME = "result"

# Nombres de columna esperados según el README. AJUSTA AQUÍ si al inspeccionar
# las columnas reales (paso 1 de main()) no coinciden.
COL_PACIENTE_ID = "paciente_id"
COL_CASO_ID = "caso_id"
COL_TRAYECTORIA_ID = "trayectoria_id"
COL_CAPA = "capa"
COL_LABEL = "label_ground_truth"
COL_DIALOGO_ID = "dialogo_id"

JSON_CELL_COLUMNS = ["comorbilidades", "adaptation_fields"]


def load_sheet(path: Path, sheet_name: str = SHEET_NAME) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No encuentro {path}. Verifica --dataset-dir o que ya tengas "
            f"el dataset del reto descargado en esa carpeta."
        )
    return pd.read_excel(path, sheet_name=sheet_name)


def inspect_columns(name: str, df: pd.DataFrame) -> None:
    print(f"\n--- {name} ---")
    print(f"  filas: {len(df)}  columnas: {list(df.columns)}")


def parse_json_cell(value):
    """Convierte una celda que contiene una lista JSON en texto a una lista de Python."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        # Deja constancia en vez de reventar: mejor ver el dato crudo que perder la fila.
        return {"_raw_unparsed": value}


def build_caso_id_from_trayectoria(df_trayectorias: pd.DataFrame) -> pd.DataFrame:
    """Agrega caso_id = 'caso_' + trayectoria_id si no viene ya en el archivo."""
    if COL_CASO_ID not in df_trayectorias.columns:
        if COL_TRAYECTORIA_ID not in df_trayectorias.columns:
            raise KeyError(
                f"No encuentro ni '{COL_CASO_ID}' ni '{COL_TRAYECTORIA_ID}' en "
                f"trayectorias_postop_silver.xlsx. Columnas disponibles: "
                f"{list(df_trayectorias.columns)}. Ajusta COL_TRAYECTORIA_ID arriba."
            )
        df_trayectorias = df_trayectorias.copy()
        df_trayectorias[COL_CASO_ID] = "caso_" + df_trayectorias[COL_TRAYECTORIA_ID].astype(str)
    return df_trayectorias


def validate_label_consistency(df_conversaciones: pd.DataFrame) -> None:
    """label_ground_truth debe ser constante dentro de cada caso_id (según el README)."""
    if COL_CASO_ID not in df_conversaciones.columns or COL_LABEL not in df_conversaciones.columns:
        print(
            f"  [aviso] No puedo validar consistencia de {COL_LABEL}: falta "
            f"'{COL_CASO_ID}' o '{COL_LABEL}' en dataset_final.xlsx."
        )
        return
    n_labels_por_caso = df_conversaciones.groupby(COL_CASO_ID)[COL_LABEL].nunique()
    inconsistentes = n_labels_por_caso[n_labels_por_caso > 1]
    if len(inconsistentes) > 0:
        print(f"  [ALERTA] {len(inconsistentes)} caso_id con más de un label_ground_truth distinto:")
        print(f"  {list(inconsistentes.index)}")
    else:
        print(f"  OK: label_ground_truth es constante dentro de cada caso_id.")


def print_class_balance(df_conversaciones: pd.DataFrame) -> None:
    if COL_CASO_ID not in df_conversaciones.columns or COL_LABEL not in df_conversaciones.columns:
        return
    por_caso = df_conversaciones.drop_duplicates(subset=[COL_CASO_ID])[COL_LABEL].value_counts()
    print("\n--- Balance de clases (por caso_id, no por turno) ---")
    print(por_caso)
    print("  Esperado según README: 123 verde / 25 amarillo / 12 rojo (de 160 casos)")


def filter_by_capa(df_conversaciones: pd.DataFrame, capa: str) -> pd.DataFrame:
    """
    Filtra por capa ANTES de reconstruir cualquier conversación, como advierte
    el README: un mismo caso_id contiene ambas capas mezcladas.
    """
    if COL_CAPA not in df_conversaciones.columns:
        raise KeyError(
            f"No encuentro la columna '{COL_CAPA}' en dataset_final.xlsx. "
            f"Columnas disponibles: {list(df_conversaciones.columns)}."
        )
    return df_conversaciones[df_conversaciones[COL_CAPA] == capa].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("./dataset"))
    args = parser.parse_args()

    d = args.dataset_dir

    print("=" * 70)
    print("1) INSPECCIÓN DE COLUMNAS — revisa que calcen con lo que se espera")
    print("=" * 70)

    df_conversaciones = load_sheet(d / "dataset_final.xlsx")
    df_trayectorias = load_sheet(d / "trayectorias_postop_silver.xlsx")
    df_perfil_clinico = load_sheet(d / "perfiles_clinicos_pacientes_silver_contest.xlsx")
    df_perfil_demografico = load_sheet(d / "perfiles_pacientes_co.xlsx")

    inspect_columns("dataset_final.xlsx (conversaciones, 1 fila = 1 turno)", df_conversaciones)
    inspect_columns("trayectorias_postop_silver.xlsx (cuadro clínico por caso)", df_trayectorias)
    inspect_columns("perfiles_clinicos_pacientes_silver_contest.xlsx (por paciente)", df_perfil_clinico)
    inspect_columns("perfiles_pacientes_co.xlsx (demografía, por paciente)", df_perfil_demografico)

    print("\n" + "=" * 70)
    print("2) PARSEO DE CAMPOS JSON-EN-CELDA")
    print("=" * 70)
    for col in JSON_CELL_COLUMNS:
        if col in df_perfil_clinico.columns:
            df_perfil_clinico[col] = df_perfil_clinico[col].apply(parse_json_cell)
            print(f"  Parseado '{col}' en perfiles_clinicos_pacientes_silver_contest.xlsx")
        if col in df_perfil_demografico.columns:
            df_perfil_demografico[col] = df_perfil_demografico[col].apply(parse_json_cell)
            print(f"  Parseado '{col}' en perfiles_pacientes_co.xlsx")

    print("\n" + "=" * 70)
    print("3) JOIN: paciente_id une los 4 archivos; caso_id = 'caso_' + trayectoria_id")
    print("=" * 70)
    df_trayectorias = build_caso_id_from_trayectoria(df_trayectorias)
    print(f"  Ejemplo de caso_id construidos: {df_trayectorias[COL_CASO_ID].head(3).tolist()}")

    print("\n" + "=" * 70)
    print("4) VALIDACIONES DE CONSISTENCIA")
    print("=" * 70)
    validate_label_consistency(df_conversaciones)
    print_class_balance(df_conversaciones)

    print("\n" + "=" * 70)
    print("5) FILTRADO POR CAPA (ejemplo con capa1_limpia)")
    print("=" * 70)
    try:
        df_capa1 = filter_by_capa(df_conversaciones, "capa1_limpia")
        print(f"  Turnos en capa1_limpia: {len(df_capa1)} (de {len(df_conversaciones)} totales)")
    except KeyError as e:
        print(f"  [aviso] {e}")

    print("\nListo. Si algún nombre de columna no calzó, ajústalo en las")
    print("constantes COL_* al inicio del archivo y vuelve a correr.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
