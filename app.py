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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* === FONDO PRINCIPAL === */
    .stApp {
        background: linear-gradient(135deg, #f0f7f0 0%, #e8f5e9 30%, #f1f8e9 70%, #fff8e1 100%);
    }
    
    .main > div {
        padding-top: 1rem;
    }
    
    /* === HEADER === */
    .header-container {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 40%, #388e3c 70%, #43a047 100%);
        padding: 2rem 2.5rem;
        border-radius: 0 0 28px 28px;
        margin: -3rem -3rem 1.5rem -3rem;
        box-shadow: 0 6px 24px rgba(27, 94, 32, 0.25);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        pointer-events: none;
    }
    
    .header-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: white;
        margin: 0;
        line-height: 1.2;
        position: relative;
        z-index: 1;
    }
    
    .header-subtitle {
        font-size: 0.9rem;
        font-weight: 400;
        color: rgba(255, 255, 255, 0.8);
        margin-top: 0.35rem;
        position: relative;
        z-index: 1;
    }
    
    /* === SIDEBAR === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    section[data-testid="stSidebar"] .stApp { background: transparent; }
    
    .sidebar-logo {
        font-size: 1.4rem;
        font-weight: 800;
        color: #81c784;
        text-align: center;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sidebar-section-title {
        font-size: 0.65rem;
        font-weight: 700;
        color: rgba(255, 255, 255, 0.35);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 1.25rem 0 0.6rem 0;
        padding: 0 0.5rem;
    }
    
    .sidebar-metric {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.65rem 1rem;
        margin-bottom: 0.4rem;
    }
    
    .sidebar-metric-label {
        font-size: 0.7rem;
        color: rgba(255, 255, 255, 0.45);
        margin-bottom: 0.15rem;
    }
    
    .sidebar-metric-value {
        font-size: 1rem;
        font-weight: 700;
        color: #e8f5e9;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.25rem 0.7rem;
        border-radius: 100px;
        font-size: 0.7rem;
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
    
    /* === CHAT - BURBUJAS === */
    .chat-container {
        max-width: 100%;
        margin: 0 auto;
    }
    
    .bubble-user {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
        border: 1px solid #c8e6c9;
        border-radius: 18px 18px 4px 18px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        color: #1a1a2e;
        font-size: 0.9rem;
        line-height: 1.5;
        animation: fadeIn 0.3s ease-out;
    }
    
    .bubble-assistant {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 18px 18px 18px 4px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
        line-height: 1.7;
        animation: fadeIn 0.3s ease-out;
    }
    
    .bubble-assistant p {
        margin: 0 0 0.75rem 0;
    }
    
    .bubble-assistant p:last-child {
        margin-bottom: 0;
    }
    
    .bubble-label {
        font-size: 0.7rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    .bubble-label-user {
        color: #2e7d32;
    }
    
    .bubble-label-assistant {
        color: #1565c0;
    }
    
    /* === INPUT AREA === */
    .input-area {
        background: white;
        border-radius: 20px;
        padding: 0.75rem;
        box-shadow: 0 -2px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid #e8e8e8;
        margin-top: 1rem;
    }
    
    .stTextArea textarea {
        border-radius: 14px !important;
        border: 2px solid #e0e0e0 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        background: #fafafa !important;
        padding: 0.75rem 1rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #2e7d32 !important;
        box-shadow: 0 0 0 3px rgba(46, 125, 50, 0.1) !important;
        background: white !important;
    }
    
    /* === BOTÓN ENVIAR === */
    div[data-testid="column"] .stButton button {
        background: linear-gradient(135deg, #2e7d32 0%, #388e3c 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.25) !important;
        width: 100% !important;
    }
    
    div[data-testid="column"] .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.35) !important;
    }
    
    /* === BOTONES SIDEBAR === */
    div[data-testid="stSidebar"] .stButton button {
        background: rgba(255, 255, 255, 0.06) !important;
        color: rgba(255, 255, 255, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.82rem !important;
        font-weight: 400 !important;
        text-align: left !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
        margin-bottom: 0.3rem !important;
    }
    
    div[data-testid="stSidebar"] .stButton button:hover {
        background: rgba(129, 199, 132, 0.15) !important;
        border-color: rgba(129, 199, 132, 0.3) !important;
        color: white !important;
        transform: translateX(3px) !important;
    }
    
    /* === FUENTES === */
    .fuentes-container {
        background: #f8fbf8;
        border-radius: 16px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        border: 1px solid #e8f5e9;
    }
    
    .fuentes-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #558b2f;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.75rem;
    }
    
    /* === WELCOME === */
    .welcome-card {
        background: white;
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: 0 2px 20px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(0, 0, 0, 0.04);
        margin: 2rem 0;
    }
    
    .welcome-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .welcome-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    
    .welcome-text {
        font-size: 0.9rem;
        color: #757575;
        line-height: 1.6;
        max-width: 400px;
        margin: 0 auto;
    }
    
    /* === ANIMACIONES === */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* === FOOTER === */
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1.5rem;
        color: rgba(0, 0, 0, 0.3);
        font-size: 0.75rem;
    }
    
    /* === OCULTAR ELEMENTOS POR DEFECTO === */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* === RESPONSIVE === */
    @media (max-width: 768px) {
        .header-container {
            padding: 1.25rem 1.5rem;
            margin: -2rem -1rem 1rem -1rem;
            border-radius: 0 0 20px 20px;
        }
        .header-title { font-size: 1.4rem; }
    }
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

    st.markdown('<div class="fuentes-container">', unsafe_allow_html=True)
    st.markdown('<p class="fuentes-title">📖 Fuentes consultadas</p>', unsafe_allow_html=True)
    
    for indice, citacion in enumerate(citaciones[:3], start=1):
        pagina = f" · pág. {citacion.pagina}" if citacion.pagina is not None else ""
        fuente = citacion.fuente or "Documento técnico"
        with st.expander(f"📄 {fuente}{pagina}", expanded=False):
            st.markdown(f'<div style="font-size:0.85rem; line-height:1.6; color:#424242;">{citacion.contenido}</div>', unsafe_allow_html=True)
    
    if len(citaciones) > 3:
        st.markdown(f'<p style="text-align:center;font-size:0.75rem;color:#9e9e9e;margin-top:0.5rem;">+ {len(citaciones) - 3} fuente(s) adicional(es)</p>', unsafe_allow_html=True)
    
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
# INICIALIZACIÓN DEL ESTADO DE SESIÓN
# ============================================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ultima_pregunta" not in st.session_state:
    st.session_state.ultima_pregunta = ""


# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="header-container">
    <div class="header-title">🌱 AgronomIA</div>
    <div class="header-subtitle">Asistente técnico inteligente para el cultivo de cannabis medicinal</div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# INICIALIZACIÓN DEL AGENTE
# ============================================================================

try:
    grafo, total_documentos = inicializar_agente()
except FileNotFoundError:
    st.error("### 📁 No se encontraron documentos\n\nAsegúrate de haber subido los archivos PDF a la carpeta `docs/` del repositorio.")
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
    
    boton_clickeado = None
    if st.button("🌿 ¿Qué producto controla Botrytis?", use_container_width=True, key="q1"):
        boton_clickeado = "🌿 ¿Qué producto controla Botrytis?"
    if st.button("🧪 ¿Cuál es la dosis para control de plagas?", use_container_width=True, key="q2"):
        boton_clickeado = "🧪 ¿Cuál es la dosis para control de plagas?"
    if st.button("🔬 ¿Cómo identificar síntomas de enfermedad foliar?", use_container_width=True, key="q3"):
        boton_clickeado = "🔬 ¿Cómo identificar síntomas de enfermedad foliar?"
    if st.button("🍄 ¿Qué fungicidas recomiendas para cannabis?", use_container_width=True, key="q4"):
        boton_clickeado = "🍄 ¿Qué fungicidas recomiendas para cannabis?"
    
    # --- Acerca de ---
    st.markdown('<p class="sidebar-section-title">Acerca de</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(255,255,255,0.04); border-radius: 12px; padding: 0.75rem; margin-top: 0.25rem;">
        <p style="font-size:0.75rem; color:rgba(255,255,255,0.4); line-height:1.5; margin:0;">
            <strong style="color:rgba(255,255,255,0.6);">AgronomIA</strong> usa 
            <strong style="color:#81c784;">RAG</strong> para responder preguntas 
            técnicas basándose en documentos PDF especializados.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# ÁREA PRINCIPAL - CHAT
# ============================================================================

# Procesar clic en botón rápido del sidebar
if boton_clickeado:
    # Extraer solo el texto sin el emoji
    pregunta_texto = boton_clickeado.split(" ", 1)[1] if " " in boton_clickeado else boton_clickeado
    st.session_state.ultima_pregunta = pregunta_texto
    
    # Ejecutar la consulta directamente
    resultado = consultar(pregunta_texto, grafo)
    if resultado:
        st.session_state.chat_history.append({
            "tipo": "usuario",
            "contenido": pregunta_texto
        })
        st.session_state.chat_history.append({
            "tipo": "asistente",
            "contenido": resultado.get("respuesta", "No lo sé"),
            "citaciones": resultado.get("citaciones") or []
        })

# Mostrar historial del chat
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        # Pantalla de bienvenida
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">🌱</div>
            <div class="welcome-title">Bienvenido a AgronomIA</div>
            <div class="welcome-text">
                Soy tu asistente técnico especializado en cultivo de cannabis medicinal. 
                Pregúntame sobre plagas, enfermedades, productos y dosis recomendadas.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for mensaje in st.session_state.chat_history:
            if mensaje["tipo"] == "usuario":
                st.markdown(f"""
                <div class="bubble-user">
                    <div class="bubble-label bubble-label-user">👤 Tú</div>
                    {mensaje["contenido"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bubble-assistant">
                    <div class="bubble-label bubble-label-assistant">🌱 AgronomIA</div>
                    {mensaje["contenido"].replace(chr(10), "<br>")}
                </div>
                """, unsafe_allow_html=True)
                
                if mensaje.get("citaciones"):
                    mostrar_citaciones(mensaje["citaciones"])


# ============================================================================
# INPUT - SIEMPRE VISIBLE ABAJO
# ============================================================================

st.markdown('<div class="input-area">', unsafe_allow_html=True)

pregunta = st.text_area(
    label="",
    key="input_pregunta",
    height=70,
    placeholder="Escribe tu pregunta técnica aquí...",
    label_visibility="collapsed",
)

col_input, col_btn = st.columns([4, 1])
with col_btn:
    enviar = st.button("Enviar", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# PROCESAR ENVÍO DESDE EL INPUT
# ============================================================================

if enviar and pregunta.strip():
    pregunta_texto = pregunta.strip()
    st.session_state.ultima_pregunta = pregunta_texto
    
    # Ejecutar consulta
    resultado = consultar(pregunta_texto, grafo)
    if resultado:
        st.session_state.chat_history.append({
            "tipo": "usuario",
            "contenido": pregunta_texto
        })
        st.session_state.chat_history.append({
            "tipo": "asistente",
            "contenido": resultado.get("respuesta", "No lo sé"),
            "citaciones": resultado.get("citaciones") or []
        })
    
    # Limpiar el input y recargar
    st.rerun()
    
elif enviar and not pregunta.strip():
    st.warning("✏️ Por favor, escribe una pregunta.")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="footer">
    🌱 AgronomIA · Asistente Técnico con RAG
</div>
""", unsafe_allow_html=True)