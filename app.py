# -*- coding: utf-8 -*-
"""Interfaz Streamlit para AgronomIA - Diseño Moderno."""

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

# ============================================================================
# CONFIGURACIÓN DE ESTILO
# ============================================================================

st.set_page_config(
    page_title="AgronomIA - Asistente Técnico Agrícola",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded",
)

# CSS personalizado para diseño moderno
st.markdown("""
<style>
    /* === FUENTES Y GLOBAL === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* === FONDO PRINCIPAL === */
    .stApp {
        background: linear-gradient(135deg, #f0f7f0 0%, #e8f5e9 30%, #f1f8e9 70%, #fff8e1 100%);
    }
    
    /* === HEADER === */
    .header-container {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 40%, #388e3c 70%, #43a047 100%);
        padding: 2.5rem 3rem;
        border-radius: 0 0 32px 32px;
        margin: -4rem -4rem 2rem -4rem;
        box-shadow: 0 8px 32px rgba(27, 94, 32, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        pointer-events: none;
    }
    
    .header-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        line-height: 1.2;
        position: relative;
        z-index: 1;
    }
    
    .header-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        font-weight: 400;
        color: rgba(255, 255, 255, 0.85);
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 0.35rem 1rem;
        border-radius: 100px;
        font-size: 0.8rem;
        color: white;
        margin-top: 1rem;
        position: relative;
        z-index: 1;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* === SIDEBAR === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    section[data-testid="stSidebar"] .stApp {
        background: transparent;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
        padding: 1.5rem 1rem;
    }
    
    .sidebar-logo {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        color: #81c784;
        text-align: center;
        padding: 0.5rem;
        margin-bottom: 1rem;
        letter-spacing: -0.5px;
    }
    
    .sidebar-section-title {
        font-size: 0.7rem;
        font-weight: 700;
        color: rgba(255, 255, 255, 0.4);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 1.5rem 0 0.75rem 0;
        padding: 0 0.5rem;
    }
    
    /* === MÉTRICAS DEL SIDEBAR === */
    .sidebar-metric {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-metric-label {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.5);
        margin-bottom: 0.25rem;
    }
    
    .sidebar-metric-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e8f5e9;
    }
    
    /* === BOTONES DE EJEMPLO === */
    .ejemplo-btn {
        width: 100%;
        text-align: left;
        padding: 0.7rem 1rem;
        margin-bottom: 0.4rem;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .ejemplo-btn:hover {
        background: rgba(129, 199, 132, 0.15);
        border-color: rgba(129, 199, 132, 0.3);
        color: white;
        transform: translateX(3px);
    }
    
    /* === CARD DE CONSULTA === */
    .consulta-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 2px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(0, 0, 0, 0.04);
        margin: 1.5rem 0;
    }
    
    /* === RESPUESTA === */
    .respuesta-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 24px rgba(27, 94, 32, 0.08);
        border: 1px solid rgba(27, 94, 32, 0.1);
        margin: 1.5rem 0;
        border-left: 4px solid #2e7d32;
    }
    
    .respuesta-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #2e7d32;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .respuesta-texto {
        font-size: 1rem;
        line-height: 1.7;
        color: #1a1a2e;
    }
    
    /* === FUENTES === */
    .fuentes-container {
        background: #f8fbf8;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 1px solid #e8f5e9;
    }
    
    .fuentes-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #558b2f;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }
    
    /* === STATUS BADGE === */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.8rem;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .status-online {
        background: rgba(129, 199, 132, 0.15);
        color: #2e7d32;
        border: 1px solid rgba(129, 199, 132, 0.3);
    }
    
    .status-offline {
        background: rgba(255, 183, 77, 0.15);
        color: #e65100;
        border: 1px solid rgba(255, 183, 77, 0.3);
    }
    
    /* === INPUT === */
    .stTextArea textarea {
        border-radius: 14px !important;
        border: 2px solid #e0e0e0 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        background: #fafafa !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #2e7d32 !important;
        box-shadow: 0 0 0 3px rgba(46, 125, 50, 0.1) !important;
        background: white !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #9e9e9e !important;
    }
    
    /* === BOTÓN PRINCIPAL === */
    .stButton button {
        background: linear-gradient(135deg, #2e7d32 0%, #388e3c 50%, #43a047 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 16px rgba(46, 125, 50, 0.3) !important;
        letter-spacing: 0.3px !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(46, 125, 50, 0.4) !important;
    }
    
    .stButton button:active {
        transform: translateY(0) !important;
    }
    
    /* === FOOTER === */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        color: rgba(0, 0, 0, 0.35);
        font-size: 0.8rem;
        border-top: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* === DIVIDER === */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 0, 0, 0.08), transparent);
        margin: 1.5rem 0;
    }
    
    /* === SPINNER === */
    .stSpinner > div {
        border-color: #2e7d32 !important;
    }
    
    /* === RESPONSIVE === */
    @media (max-width: 768px) {
        .header-container {
            padding: 1.5rem;
            margin: -2rem -1rem 1rem -1rem;
            border-radius: 0 0 20px 20px;
        }
        .header-title {
            font-size: 1.8rem;
        }
    }
    
    /* === ANIMACIONES === */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .respuesta-container {
        animation: fadeInUp 0.4s ease-out;
    }
    
    /* === OCULTAR ELEMENTOS POR DEFECTO DE STREAMLIT === */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FUNCIONES DE LA INTERFAZ
# ============================================================================

@st.cache_resource(show_spinner=False)
def inicializar_agente():
    """Inicializa el agente: carga PDFs, crea retriever y compila el workflow."""
    with st.spinner("🌱 Preparando base de conocimiento documental..."):
        documentos = cargar_documentos(DOCS_DIR)
        retriever = crear_retriever(documentos)
        grafo = crear_workflow(retriever)
    return grafo, len(documentos)


def mostrar_citaciones(documentos: list) -> None:
    """Muestra las fuentes usadas para generar la respuesta."""
    citaciones = extraer_citaciones(documentos)
    if not citaciones:
        return

    with st.container():
        st.markdown('<div class="fuentes-container">', unsafe_allow_html=True)
        st.markdown('<p class="fuentes-title">📖 Fuentes consultadas</p>', unsafe_allow_html=True)
        
        for indice, citacion in enumerate(citaciones[:3], start=1):  # Mostrar máximo 3
            pagina = f" · pág. {citacion.pagina}" if citacion.pagina is not None else ""
            fuente = citacion.fuente or "Documento técnico"
            with st.expander(f"📄 {fuente}{pagina}", expanded=False):
                st.markdown(f'<div style="font-size:0.9rem; line-height:1.6; color:#424242;">{citacion.contenido}</div>', unsafe_allow_html=True)
        
        if len(citaciones) > 3:
            st.markdown(f'<p style="text-align:center;font-size:0.8rem;color:#9e9e9e;margin-top:0.5rem;">+ {len(citaciones) - 3} fuente(s) adicional(es)</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


def consultar(pregunta: str, grafo) -> dict | None:
    """Valida la pregunta y ejecuta la consulta contra el grafo."""
    error = validar_pregunta(pregunta)
    if error:
        st.warning(error)
        return None

    with st.spinner("🔍 Analizando documentos técnicos..."):
        return ejecutar_consulta(pregunta, grafo)


# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="header-container">
    <div class="header-title">🌱 AgronomIA</div>
    <div class="header-subtitle">Asistente técnico inteligente para el cultivo de cannabis medicinal</div>
    <div class="header-badge">
        <span>⚡</span> 
        <span>Potenciado con RAG + LangGraph</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# INICIALIZACIÓN DEL AGENTE
# ============================================================================

try:
    grafo, total_documentos = inicializar_agente()
    init_ok = True
except FileNotFoundError:
    st.error("""
        ### 📁 No se encontraron documentos
        
        Asegúrate de haber subido los archivos PDF a la carpeta `docs/` del repositorio.
    """)
    st.stop()
    init_ok = False
except ImportError as e:
    st.error(f"### ❌ Dependencia faltante\n\n```\n{e}\n```")
    st.stop()
except Exception as exc:
    st.error(f"### ❌ Error de inicialización\n\n```\n{exc}\n```")
    st.stop()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown('<p class="sidebar-logo">🌱 AgronomIA</p>', unsafe_allow_html=True)
    
    # --- Estado del sistema ---
    st.markdown('<p class="sidebar-section-title">Estado del sistema</p>', unsafe_allow_html=True)
    
    groq_disponible = os.getenv("GROQ_API_KEY") is not None
    
    st.markdown(f"""
    <div class="sidebar-metric">
        <div class="sidebar-metric-label">📄 Documentos cargados</div>
        <div class="sidebar-metric-value">{total_documentos} páginas</div>
    </div>
    """, unsafe_allow_html=True)
    
    if groq_disponible:
        st.markdown("""
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">🤖 Motor de IA</div>
            <div><span class="status-badge status-online">● Groq LLM Activo</span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">🤖 Motor de IA</div>
            <div><span class="status-badge status-offline">○ Modo local</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    # --- Consultas rápidas ---
    st.markdown('<p class="sidebar-section-title">Consultas rápidas</p>', unsafe_allow_html=True)
    
    ejemplos = [
        "🌿 ¿Qué producto controla Botrytis?",
        "🧪 ¿Cuál es la dosis para control de plagas?",
        "🔬 ¿Cómo identificar síntomas de enfermedad foliar?",
        "🍄 ¿Qué fungicidas recomiendas para cannabis?",
    ]
    
    for ejemplo in ejemplos:
        if st.button(ejemplo, use_container_width=True, key=f"btn_{ejemplo[:10]}"):
            st.session_state["pregunta"] = ejemplo
    
    # --- Info ---
    st.markdown('<p class="sidebar-section-title">Acerca de</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem; margin-top: 0.5rem;">
        <p style="font-size:0.8rem; color:rgba(255,255,255,0.5); line-height:1.5; margin:0;">
            <strong style="color:rgba(255,255,255,0.7);">AgronomIA</strong> utiliza 
            <strong style="color:#81c784;">RAG</strong> (Retrieval Augmented Generation) 
            para responder preguntas técnicas basándose en documentos PDF especializados.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# ÁREA PRINCIPAL - CONSULTA
# ============================================================================

st.markdown('<div class="consulta-card">', unsafe_allow_html=True)

st.markdown("""
<p style="font-size:0.9rem; font-weight:600; color:#424242; margin-bottom:0.5rem;">
    💬 ¿En qué puedo ayudarte hoy?
</p>
""", unsafe_allow_html=True)

pregunta = st.text_area(
    label="",
    key="pregunta",
    height=100,
    placeholder="Ej: ¿Qué producto controla la Botrytis en cultivos de cannabis? ¿Cuál es su dosis recomendada?",
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    consultar_click = st.button("🔍 Consultar documentos", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# RESULTADOS
# ============================================================================

if consultar_click and pregunta.strip():
    resultado = consultar(pregunta.strip(), grafo)
    if resultado:
        respuesta = resultado.get("respuesta", "No lo sé")
        
        # Mostrar respuesta
        st.markdown("""
        <div class="respuesta-container">
            <div class="respuesta-label">💬 Respuesta</div>
            <div class="respuesta-texto">{}</div>
        </div>
        """.format(respuesta.replace("\n", "<br>")), unsafe_allow_html=True)
        
        # Mostrar fuentes
        mostrar_citaciones(resultado.get("citaciones") or [])
        
elif consultar_click and not pregunta.strip():
    st.warning("✏️ Por favor, escribe una pregunta antes de consultar.")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="footer">
    🌱 AgronomIA — Asistente Técnico Agrícola con RAG<br>
    <span style="font-size:0.75rem;">Desarrollado con LangChain · LangGraph · FAISS · Groq · Streamlit</span>
</div>
""", unsafe_allow_html=True)