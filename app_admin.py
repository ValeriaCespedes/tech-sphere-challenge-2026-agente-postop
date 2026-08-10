import streamlit as st
from src.ingesta import indexar_documento, listar_documentos, eliminar_documento

st.set_page_config(page_title="Consola de administración — Conocimiento clínico", layout="wide")
st.title("📋 Consola de administración del conocimiento clínico")
st.caption("Sube, lista y elimina documentos que alimentan el RAG del agente. Los cambios son inmediatos.")

st.divider()

# --- Subir documento ---
st.subheader("Subir documento")
archivo_subido = st.file_uploader("Selecciona un PDF", type=["pdf"])

if archivo_subido is not None:
    if st.button("Procesar y agregar al conocimiento"):
        with st.spinner(f"Procesando {archivo_subido.name}..."):
            resultado = indexar_documento(
                pdf_bytes=archivo_subido.read(),
                nombre_archivo=archivo_subido.name,
            )
        if resultado["ok"]:
            st.success(f"✅ Procesado y disponible — {resultado['chunks']} fragmentos indexados.")
        else:
            st.error(f"❌ No se pudo procesar: {resultado['motivo']}")

st.divider()

# --- Listar documentos ---
st.subheader("Documentos cargados")
documentos = listar_documentos()

if not documentos:
    st.info("No hay documentos indexados todavía.")
else:
    for doc in documentos:
        col1, col2, col3 = st.columns([5, 2, 1])
        with col1:
            st.write(f"📄 {doc['archivo']}")
        with col2:
            st.write(f"{doc['chunks']} fragmentos")
        with col3:
            if st.button("Eliminar", key=f"del_{doc['archivo']}"):
                borrados = eliminar_documento(doc["archivo"])
                st.success(f"Eliminado ({borrados} fragmentos). El agente ya no tiene acceso a este documento.")
                st.rerun()