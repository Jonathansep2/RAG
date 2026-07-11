# -*- coding: utf-8 -*-
"""Interfaz Streamlit para AgronomIA."""

from __future__ import annotations

import os

import streamlit as st


def configurar_secretos() -> None:
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        groq_api_key = None

    if groq_api_key and not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = groq_api_key


configurar_secretos()

from src.rag_agente import (
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
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #f5faf5, #eef6ee 50%, #f7faf0); }
    .main > div { padding: 1rem 0; }

    /* HEADER */
    .header {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        padding: 1.5rem 2rem;
        border-radius: 0 0 24px 24px;
        margin: -2rem -2rem 1.5rem -2rem;
        box-shadow: 0 4px 20px rgba(27,94,32,0.2);
    }
    .header h1 { font-size: 1.5rem; font-weight: 800; color: white; margin: 0; }
    .header p { font-size: 0.85rem; color: rgba(255,255,255,0.8); margin: 0.25rem 0 0 0; }

    /* SIDEBAR */
    section[data-testid="stSidebar"] { background: #1a1a2e; }
    .sb-logo { font-size: 1.3rem; font-weight: 800; color: #81c784; text-align: center; padding: 1rem 0.5rem; }
    .sb-sec {
        font-size: 0.6rem; font-weight: 700; color: rgba(255,255,255,0.35);
        text-transform: uppercase; letter-spacing: 1.5px; margin: 1.5rem 0 0.5rem 0; padding: 0 0.5rem;
    }
    .sb-card {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px; padding: 0.6rem 0.9rem; margin: 0 0.5rem 0.4rem 0.5rem;
    }
    .sb-lbl { font-size: 0.65rem; color: rgba(255,255,255,0.45); margin-bottom: 0.15rem; }
    .sb-val { font-size: 0.9rem; font-weight: 700; color: #e8f5e9; }
    .bdg {
        display: inline-block; padding: 0.2rem 0.6rem; border-radius: 100px;
        font-size: 0.6rem; font-weight: 600;
    }
    .bdg-on { background: rgba(129,199,132,0.15); color: #66bb6a; border: 1px solid rgba(129,199,132,0.3); }
    .bdg-off { background: rgba(255,183,77,0.15); color: #ff9800; border: 1px solid rgba(255,183,77,0.3); }
    
    div[data-testid="stSidebar"] .stButton button {
        background: rgba(255,255,255,0.06) !important; color: rgba(255,255,255,0.8) !important;
        border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 10px !important;
        padding: 0.5rem 1rem !important; font-size: 0.78rem !important; text-align: left !important;
        box-shadow: none !important; margin: 0 0.5rem 0.25rem 0.5rem !important; width: calc(100% - 1rem) !important;
    }
    div[data-testid="stSidebar"] .stButton button:hover {
        background: rgba(129,199,132,0.15) !important;
        border-color: rgba(129,199,132,0.3) !important; color: white !important;
    }

    /* INPUT */
    .card {
        background: white; border-radius: 16px; padding: 1.25rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04); border: 1px solid #eee;
        margin: 1rem 0;
    }
    .stTextArea textarea {
        border-radius: 12px !important; border: 2px solid #e0e0e0 !important;
        font-size: 0.9rem !important; background: #fafafa !important;
        padding: 0.7rem 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #2e7d32 !important;
        box-shadow: 0 0 0 3px rgba(46,125,50,0.1) !important;
        background: white !important;
    }
    .stButton button {
        background: linear-gradient(135deg, #2e7d32, #388e3c) !important; color: white !important;
        border: none !important; border-radius: 12px !important;
        padding: 0.55rem 2rem !important; font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(46,125,50,0.25) !important;
    }
    .stButton button:hover { transform: translateY(-1px) !important; }

    /* RESPUESTA */
    .resp {
        background: white; border-radius: 14px; padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(27,94,32,0.06);
        border: 1px solid rgba(27,94,32,0.1);
        border-left: 4px solid #2e7d32;
        margin: 1.5rem 0; line-height: 1.7;
    }
    .resp-lbl {
        font-size: 0.65rem; font-weight: 700; color: #2e7d32;
        text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.75rem;
    }
    .fuentes {
        background: #f8fbf8; border-radius: 12px;
        padding: 1rem 1.25rem; margin: 1rem 0;
        border: 1px solid #e8f5e9;
    }
    .ftitle {
        font-size: 0.65rem; font-weight: 700; color: #558b2f;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;
    }
    @keyframes fade { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    .resp { animation: fade 0.3s ease-out; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def inicializar_agente():
    with st.spinner("🌱 Preparando base documental..."):
        docs = cargar_documentos(DOCS_DIR)
        retriever = crear_retriever(docs)
        grafo = crear_workflow(retriever)
    return grafo, len(docs)


def mostrar_fuentes(documentos: list) -> None:
    citas = extraer_citaciones(documentos)
    if not citas:
        return
    st.markdown('<div class="fuentes">', unsafe_allow_html=True)
    st.markdown('<div class="ftitle">📖 Fuentes consultadas</div>', unsafe_allow_html=True)
    for i, c in enumerate(citas[:4], 1):
        pag = f" · pág. {c.pagina}" if c.pagina is not None else ""
        src = c.fuente or "Documento"
        with st.expander(f"{i}. {src}{pag}", expanded=False):
            st.write(c.contenido)
    if len(citas) > 4:
        st.caption(f"+ {len(citas) - 4} fuente(s) más")
    st.markdown('</div>', unsafe_allow_html=True)


def ejecutar(pregunta: str, grafo) -> dict | None:
    err = validar_pregunta(pregunta)
    if err:
        st.warning(err)
        return None
    with st.spinner("🔍 Consultando documentos técnicos..."):
        return ejecutar_consulta(pregunta, grafo)


# ============================================================================
# HEADER
# ============================================================================

st.markdown('<div class="header"><h1>🌱 AgronomIA</h1><p>Asistente técnico para cultivo de cannabis medicinal</p></div>', unsafe_allow_html=True)


# ============================================================================
# INICIALIZAR
# ============================================================================

try:
    grafo, total_docs = inicializar_agente()
except FileNotFoundError:
    st.error("No se encontraron archivos PDF en la carpeta `docs/`.")
    st.stop()
except Exception as e:
    st.error(f"Error al inicializar: {e}")
    st.stop()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown('<div class="sb-logo">🌱 AgronomIA</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Estado</div>', unsafe_allow_html=True)
    groq_ok = os.getenv("GROQ_API_KEY") is not None
    badge = '<span class="bdg bdg-on">● Groq activo</span>' if groq_ok else '<span class="bdg bdg-off">○ Modo local</span>'
    st.markdown(f'<div class="sb-card"><div class="sb-lbl">📄 Documentos</div><div class="sb-val">{total_docs} páginas</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-card"><div class="sb-lbl">🤖 Motor</div><div style="margin-top:0.2rem;">{badge}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Consultas rápidas</div>', unsafe_allow_html=True)

    preguntas = [
        "¿Qué producto controla Botrytis?",
        "¿Cuál es la dosis para control de plagas?",
        "¿Cómo identificar síntomas de enfermedad foliar?",
        "¿Qué fungicidas recomiendas para cannabis?",
    ]

    pregunta_sel = None
    for i, q in enumerate(preguntas):
        if st.button(q, key=f"qr_{i}", use_container_width=True):
            pregunta_sel = q

    st.markdown('<div class="sb-sec">Acerca de</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:0.7rem;font-size:0.7rem;color:rgba(255,255,255,0.4);line-height:1.5;margin:0 0.5rem;">'
        'AgronomIA usa <strong style="color:#81c784;">RAG</strong> para responder preguntas basadas en documentos PDF técnicos.</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# ÁREA PRINCIPAL
# ============================================================================

st.markdown('<div class="card">', unsafe_allow_html=True)

pregunta = st.text_area(
    label="",
    height=90,
    placeholder="Ej: ¿Qué producto controla la Botrytis en cannabis? ¿Cuál es su dosis?",
    key="input_q",
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 1.5, 1])
with col2:
    btn = st.button("🔍 Consultar", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# RESULTADOS
# ============================================================================

# Estado para recordar la última respuesta mostrada
if "ultima_respuesta" not in st.session_state:
    st.session_state.ultima_respuesta = None

# Si se hizo clic en botón rápido
if pregunta_sel:
    res = ejecutar(pregunta_sel, grafo)
    if res:
        st.session_state.ultima_respuesta = res

# Si se hizo clic en Consultar
elif btn and pregunta.strip():
    res = ejecutar(pregunta.strip(), grafo)
    if res:
        st.session_state.ultima_respuesta = res
elif btn and not pregunta.strip():
    st.warning("✏️ Escribe una pregunta primero.")

# Mostrar respuesta si existe
if st.session_state.ultima_respuesta:
    resp = st.session_state.ultima_respuesta
    st.markdown(
        f'<div class="resp"><div class="resp-lbl">💬 Respuesta</div>{resp.get("respuesta", "No lo sé").replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )
    mostrar_fuentes(resp.get("citaciones") or [])
