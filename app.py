# -*- coding: utf-8 -*-
"""Interfaz Streamlit para AgronomIA."""

from __future__ import annotations

import os

import streamlit as st


def configurar_secretos() -> None:
    """Carga GROQ_API_KEY desde Streamlit secrets si esta disponible."""
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        groq_api_key = None

    if groq_api_key and not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = groq_api_key


configurar_secretos()

from src.rag_agente import (  # noqa: E402
    DOCS_DIR,
    cargar_documentos,
    crear_retriever,
    crear_workflow,
    ejecutar_consulta,
    extraer_citaciones,
    validar_pregunta,
)


st.set_page_config(
    page_title="AgronomIA",
    page_icon="🌾",
    layout="wide",
)


@st.cache_resource(show_spinner="Preparando base documental...")
def inicializar_agente():
    """Inicializa el agente: carga PDFs, crea retriever y compila el workflow."""
    documentos = cargar_documentos(DOCS_DIR)
    retriever = crear_retriever(documentos)
    grafo = crear_workflow(retriever)
    return grafo, len(documentos)


def mostrar_citaciones(documentos: list) -> None:
    """Muestra las fuentes usadas para generar la respuesta."""
    citaciones = extraer_citaciones(documentos)
    if not citaciones:
        return

    st.subheader("Fuentes")
    for indice, citacion in enumerate(citaciones, start=1):
        pagina = f" · p. {citacion.pagina}" if citacion.pagina is not None else ""
        fuente = citacion.fuente or "Documento"
        with st.expander(f"{indice}. {fuente}{pagina}"):
            st.write(citacion.contenido)


def consultar(pregunta: str, grafo) -> dict | None:
    """Valida la pregunta y ejecuta la consulta contra el grafo."""
    error = validar_pregunta(pregunta)
    if error:
        st.warning(error)
        return None

    with st.spinner("Consultando documentos..."):
        return ejecutar_consulta(pregunta, grafo)


# ============================================================================
# UI
# ============================================================================

st.title("🌾 AgronomIA")
st.caption("Asistente técnico para cultivo de cannabis medicinal")

# Inicializar el agente (con caché de sesión)
try:
    grafo, total_documentos = inicializar_agente()
except FileNotFoundError:
    st.error(
        "No se encontraron archivos PDF en la carpeta `docs/`.\n\n"
        "Asegúrate de haber subido los documentos al repositorio."
    )
    st.stop()
except ImportError as e:
    st.error(f"Dependencia faltante: {e}")
    st.stop()
except Exception as exc:
    st.error(f"No fue posible inicializar el agente: {exc}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Estado")
    st.metric("Páginas cargadas", total_documentos)
    st.metric("Carpeta docs", "OK" if DOCS_DIR.exists() else "No disponible")

    groq_disponible = os.getenv("GROQ_API_KEY") is not None
    st.metric("LLM (Groq)", "Conectado" if groq_disponible else "Fallback local")

    st.divider()
    st.header("Consultas rápidas")
    ejemplos = [
        "¿Qué producto controla Botrytis?",
        "¿Cuál es la dosis recomendada para control de plagas?",
        "¿Cómo identificar síntomas de enfermedad foliar?",
    ]
    for ejemplo in ejemplos:
        if st.button(ejemplo, use_container_width=True):
            st.session_state["pregunta"] = ejemplo

    st.divider()
    st.caption(
        "**AgronomIA** usa **RAG** para responder preguntas técnicas "
        "basadas en los documentos PDF cargados en `docs/`."
    )

# Área de entrada
pregunta = st.text_area(
    "Escribe tu pregunta",
    key="pregunta",
    height=120,
    placeholder="Ej: ¿Qué producto controla Botrytis?",
)

consultar_click = st.button("Consultar", type="primary", use_container_width=True)

# Resultados
if consultar_click:
    resultado = consultar(pregunta.strip(), grafo)
    if resultado:
        triaje = resultado.get("triaje", {})

        col_decision, col_urgencia, col_accion = st.columns(3)
        col_decision.metric("Decisión", triaje.get("decision", "N/D"))
        col_urgencia.metric("Urgencia", triaje.get("urgencia", "N/D"))
        col_accion.metric("Acción final", resultado.get("accion_final", "N/D"))

        st.subheader("Respuesta")
        st.write(resultado.get("respuesta", "No lo sé"))

        mostrar_citaciones(resultado.get("citaciones") or [])