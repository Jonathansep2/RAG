# -*- coding: utf-8 -*-
"""Núcleo de AgronomIA: configuración, logging, RAG y workflow."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    FAISS = None

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_community.embeddings import FastEmbedEmbeddings
except ImportError:
    FastEmbedEmbeddings = None


def configurar_salida_utf8() -> None:
    """Evita errores y texto roto en consolas Windows con cp1252."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


configurar_salida_utf8()


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
ENV_FILE = ROOT_DIR / ".env"
LOGS_DIR = ROOT_DIR / "logs"

load_dotenv(ENV_FILE)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3  # Un poco de creatividad para mejores respuestas

# Chunks más grandes para dar contexto suficiente al LLM
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVER_K = 8  # Traer más fragmentos para cubrir mejor la respuesta
EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Detecta si estamos en Streamlit Cloud (entorno efímero sin escritura persistente)
EN_STREAMLIT_CLOUD = os.getenv("STREAMLIT_RUN_ON_SAVE") is not None or os.getenv("STREAMLIT_SERVER_BASE_URL") is not None


# ============================================================================
# LOGGING
# ============================================================================

_log_handlers: list = [logging.StreamHandler()]

if not EN_STREAMLIT_CLOUD:
    try:
        LOGS_DIR.mkdir(exist_ok=True)
        _log_handlers.append(
            logging.FileHandler(LOGS_DIR / "rag_agente.log", encoding="utf-8")
        )
    except Exception:
        pass  # Si no se puede crear el archivo de log, solo usamos consola

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=_log_handlers,
)

logger = logging.getLogger("AgronomIA")


def log_info(message: str) -> None:
    logger.info(message)


def log_warning(message: str) -> None:
    logger.warning(message)


def log_error(message: str) -> None:
    logger.error(message)


def log_debug(message: str) -> None:
    logger.debug(message)


# ============================================================================
# PROMPTS
# ============================================================================

PROMPT_TRIAJE = """Eres un especialista en triaje para un asistente técnico agrícola.
Analiza el siguiente mensaje y devuelve ÚNICAMENTE un JSON válido con este formato exacto:
{"decision": "AUTO_RESOLVER", "urgencia": "BAJA", "campos_faltantes": []}

Reglas:
- decision puede ser: "AUTO_RESOLVER" (preguntas técnicas sobre cultivo, plagas, enfermedades, productos, dosis) | "PEDIR_INFO" (preguntas imprecisas o genéricas) | "ABRIR_TICKET" (solicitudes de asesoría personalizada, urgencias)
- urgencia puede ser: "BAJA" | "MEDIANA" | "ALTA"
- campos_faltantes es un array de strings, vacío si no hay campos faltantes.

Responde SOLO con JSON válido, sin explicaciones adicionales."""

PROMPT_SYSTEM_RAG = (
    "Eres AgronomIA, un asistente técnico experto en cultivo de cannabis medicinal. "
    "Tu función es ayudar a ingenieros agrónomos y agricultores con preguntas detalladas sobre:\n"
    "- Manejo fitosanitario (plagas, enfermedades, malezas)\n"
    "- Productos agroquímicos (fungicidas, insecticidas, bactericidas, acaricidas)\n"
    "- Dosis, formas de aplicación y períodos de carencia\n"
    "- Síntomas y diagnosis de problemas en cultivos\n"
    "- Recomendaciones técnicas para el control de plagas y enfermedades\n\n"
    "Utiliza la información del contexto proporcionado para elaborar una respuesta COMPLETA y DETALLADA "
    "de al menos un párrafo. Organiza la información de forma clara: menciona el producto, "
    "su ingrediente activo, la dosis recomendada, el método de aplicación y cualquier otra "
    "información relevante del contexto.\n\n"
    "Si en el contexto hay información sobre dosis, inclúyela específicamente (por ejemplo, "
    "'1.5 L/ha' o '2 cc/L'). Si el contexto contiene datos sobre período de carencia o "
    "restricciones, menciónalos también.\n\n"
    "Si el contexto no contiene suficiente información para responder adecuadamente, responde: "
    "'No encontré suficiente información en los documentos técnicos para responder tu consulta.'"
)

PROMPT_SYSTEM_RAG_FALLBACK = (
    "Eres AgronomIA, un asistente técnico experto en cultivo de cannabis medicinal. "
    "A continuación se presentan fragmentos de documentos técnicos. Usa esta información "
    "para responder de manera COMPLETA y DETALLADA la pregunta del usuario.\n\n"
    "Fragmentos de documentos:\n{contexto}\n\n"
    "Pregunta: {pregunta}\n\n"
    "IMPORTANTE: Redacta una respuesta de al menos un párrafo completo, detallando "
    "productos, dosis, aplicaciones y cualquier información relevante."
)


# ============================================================================
# TIPOS Y VALIDACIÓN
# ============================================================================

class AgentState(TypedDict, total=False):
    """Estado compartido por los nodos del workflow."""

    pregunta: str
    triaje: dict
    respuesta: str | None
    citaciones: list | None
    documentos_encontrados: bool
    rag_exito: bool
    accion_final: str


def validar_pregunta(pregunta: str) -> Optional[str]:
    """Valida la pregunta del usuario antes de enviarla al agente."""
    if not pregunta:
        error = "La pregunta no puede estar vacía"
        log_warning(error)
        return error

    if len(pregunta) < 5:
        error = "La pregunta es muy corta (mínimo 5 caracteres)"
        log_warning(error)
        return error

    if len(pregunta) > 2000:
        error = "La pregunta es muy larga (máximo 2000 caracteres)"
        log_warning(error)
        return error

    if not any(c.isalpha() for c in pregunta):
        error = "La pregunta debe contener al menos una palabra válida"
        log_warning(error)
        return error

    return None


def validar_estado(state: dict) -> Optional[str]:
    """Valida que el estado del agente tenga la pregunta necesaria."""
    if not state or "pregunta" not in state:
        return "El estado debe contener una pregunta"
    return validar_pregunta(state["pregunta"])


# ============================================================================
# CITACIONES
# ============================================================================

@dataclass
class Citacion:
    """Representa una cita tomada de un documento recuperado."""

    contenido: str
    pagina: Optional[int] = None
    fuente: Optional[str] = None
    relevancia: Optional[float] = None

    def __str__(self) -> str:
        resultado = self.contenido[:150].replace("\n", " ")
        if self.relevancia:
            resultado += f" (relevancia: {self.relevancia:.1%})"
        if self.pagina:
            resultado += f" [p. {self.pagina}]"
        if self.fuente:
            resultado += f" - {self.fuente}"
        return resultado


def extraer_citaciones(documentos_relacionados: list) -> List[Citacion]:
    """Convierte documentos recuperados en citas legibles."""
    citaciones = []
    for doc in documentos_relacionados:
        metadata = getattr(doc, "metadata", {})
        fuente = metadata.get("source")
        if fuente:
            fuente = fuente.split("\\")[-1].split("/")[-1]

        citaciones.append(
            Citacion(
                contenido=getattr(doc, "page_content", str(doc)),
                pagina=metadata.get("page"),
                fuente=fuente,
            )
        )

    return citaciones


def formatear_citaciones(citaciones: List[Citacion], max_citaciones: int = 5) -> str:
    """Formatea citas para mostrarlas en consola."""
    if not citaciones:
        return ""

    citaciones_limitadas = citaciones[:max_citaciones]
    resultado = f"\nFuentes ({len(citaciones_limitadas)} de {len(citaciones)}):\n"

    for i, citacion in enumerate(citaciones_limitadas, start=1):
        resultado += f"   {i}. {citacion}\n"

    if len(citaciones) > max_citaciones:
        resultado += f"\n   ... y {len(citaciones) - max_citaciones} más"

    return resultado


# ============================================================================
# DOCUMENTOS Y RETRIEVER
# ============================================================================

class EmbeddingsSimples(Embeddings):
    """Fallback mínimo cuando FastEmbed no está disponible."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(sum(ord(ch) for ch in text.lower()) % 1000)] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float(sum(ord(ch) for ch in text.lower()) % 1000)]


def cargar_documentos(docs_dir: Path = DOCS_DIR) -> List:
    """Carga todos los PDFs disponibles en la carpeta docs."""
    if not docs_dir.exists():
        error = f"No existe la carpeta de documentos: {docs_dir}"
        log_error(error)
        raise FileNotFoundError(error)

    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        error = f"No se encontraron archivos PDF en: {docs_dir}"
        log_error(error)
        raise FileNotFoundError(error)

    log_info(f"Cargando {len(pdf_files)} archivo(s) PDF...")
    documentos = []
    for pdf_path in pdf_files:
        try:
            docs = PyMuPDFLoader(str(pdf_path)).load()
            documentos.extend(docs)
            log_info(f"OK {pdf_path.name} cargado ({len(docs)} páginas)")
        except Exception as exc:
            log_error(f"Error cargando {pdf_path.name}: {exc}")

    if not documentos:
        error = "No fue posible cargar ningún documento PDF."
        log_error(error)
        raise ValueError(error)

    log_info(f"Total de documentos cargados: {len(documentos)}")
    return documentos


def crear_retriever(documentos: List):
    """Crea un retriever FAISS a partir de los documentos cargados."""
    if FAISS is None:
        raise ImportError("FAISS no está disponible. Instala la dependencia faiss-cpu.")

    log_info("Dividiendo documentos en chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documentos)
    log_info(f"Total de chunks creados: {len(chunks)}")

    if FastEmbedEmbeddings is not None:
        try:
            embeddings = FastEmbedEmbeddings(model_name=EMBEDDINGS_MODEL)
            log_info("OK Usando FastEmbedEmbeddings")
        except Exception as exc:
            log_warning(f"FastEmbedEmbeddings falló al instanciar, usando embeddings simples: {exc}")
            embeddings = EmbeddingsSimples()
    else:
        log_warning("FastEmbedEmbeddings no instalado (paquete fastembed opcional), usando embeddings simples")
        embeddings = EmbeddingsSimples()

    log_info("Creando FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    log_info("OK Vectorstore creado exitosamente")

    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})


# ============================================================================
# LLM, TRIAJE Y RAG
# ============================================================================

def cargar_llm() -> Optional[object]:
    """Carga ChatGroq si existe API key y la dependencia está instalada."""
    if ChatGroq is None:
        log_warning("langchain-groq no está disponible; se usará modo fallback local.")
        return None

    if not GROQ_API_KEY:
        log_warning("GROQ_API_KEY no configurada; se usará modo fallback local.")
        return None

    return ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=GROQ_API_KEY,
    )


llm = cargar_llm()


def triaje(mensaje: str) -> Dict:
    """Clasifica la pregunta para decidir el siguiente paso del workflow."""
    error = validar_pregunta(mensaje)
    if error:
        log_error(f"Validación de pregunta fallida: {error}")
        return {
            "decision": "PEDIR_INFO",
            "urgencia": "MEDIANA",
            "campos_faltantes": [error],
        }

    log_info(f"Realizando triaje: {mensaje[:100]}...")

    if llm is None:
        return triaje_local(mensaje)

    try:
        respuesta = llm.invoke(
            [
                SystemMessage(content=PROMPT_TRIAJE),
                HumanMessage(content=mensaje),
            ]
        )
        datos = extraer_json_triaje(respuesta.content.strip())
        resultado = {
            "decision": datos.get("decision", "PEDIR_INFO"),
            "urgencia": datos.get("urgencia", "MEDIANA"),
            "campos_faltantes": datos.get("campos_faltantes", []),
        }
        log_info(f"Triaje: {resultado['decision']} - {resultado['urgencia']}")
        return resultado
    except Exception as exc:
        log_error(f"Error en triaje con LLM: {exc}")
        return triaje_local(mensaje)


def triaje_local(mensaje: str) -> Dict:
    """Fallback simple para clasificar preguntas sin LLM."""
    mensaje_low = mensaje.lower()
    palabras_ticket = [
        "recomendar",
        "asesoría",
        "asesoria",
        "ayuda",
        "urgente",
        "problema",
        "emergencia",
        "ticket",
    ]
    palabras_rag = [
        "qué",
        "que",
        "cuál",
        "cual",
        "cómo",
        "como",
        "ingrediente",
        "dosis",
        "fungicida",
        "insecticida",
        "síntoma",
        "sintoma",
        "enfermedad",
        "plaga",
        "aplicación",
        "aplicacion",
        "controla",
        "periodo",
        "período",
    ]

    if any(palabra in mensaje_low for palabra in palabras_ticket):
        decision, urgencia = "ABRIR_TICKET", "ALTA"
    elif any(palabra in mensaje_low for palabra in palabras_rag):
        decision, urgencia = "AUTO_RESOLVER", "BAJA"
    else:
        decision, urgencia = "PEDIR_INFO", "MEDIANA"

    log_info(f"Triaje (fallback): {decision} - {urgencia}")
    return {"decision": decision, "urgencia": urgencia, "campos_faltantes": []}


def extraer_json_triaje(texto: str) -> Dict:
    """Extrae el JSON de triaje incluso si el modelo agrega texto adicional."""
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", texto, re.DOTALL)
        if not match:
            log_warning("No se pudo extraer JSON del triaje")
            return {
                "decision": "PEDIR_INFO",
                "urgencia": "MEDIANA",
                "campos_faltantes": ["Error al parsear respuesta del modelo"],
            }

        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            log_warning(f"No se pudo parsear JSON del triaje: {texto[:100]}")
            return {
                "decision": "PEDIR_INFO",
                "urgencia": "MEDIANA",
                "campos_faltantes": ["Error al parsear respuesta del modelo"],
            }


def busqueda_de_respuestas_RAG(pregunta: str, retriever) -> Dict:
    """Busca contexto en los PDFs y genera una respuesta."""
    log_info(f"Buscando respuesta RAG para: {pregunta[:80]}...")
    documentos_relacionados = retriever.invoke(pregunta)

    if not documentos_relacionados:
        log_warning("No se encontraron documentos relacionados")
        return {
            "respuesta": "No encontré información en los documentos técnicos disponibles para responder tu consulta.",
            "citaciones": [],
            "documentos_encontrados": False,
        }

    log_info(f"OK {len(documentos_relacionados)} documento(s) encontrado(s)")

    if llm is None:
        answer = resumir_contexto_recuperado(documentos_relacionados, pregunta)
    else:
        try:
            contexto = "\n\n---\n\n".join(
                f"[{i+1}] {doc.page_content}" 
                for i, doc in enumerate(documentos_relacionados)
            )
            respuesta = llm.invoke(
                [
                    SystemMessage(content=PROMPT_SYSTEM_RAG),
                    HumanMessage(
                        content=f"Contexto:\n{contexto}\n\nPregunta del agrónomo: {pregunta}"
                    ),
                ]
            )
            answer = getattr(respuesta, "content", str(respuesta)).strip()
        except Exception as exc:
            log_error(f"Error generando respuesta RAG: {exc}")
            answer = resumir_contexto_recuperado(documentos_relacionados, pregunta)

    # Verificar si la respuesta es útil o es un "no sé"
    if len(answer) < 30 and ("no lo sé" in answer.lower() or "no sé" in answer.lower()):
        log_info("Respuesta insuficiente, usando resumen alternativo")
        answer = resumir_contexto_recuperado(documentos_relacionados, pregunta)

    log_info(f"OK Respuesta generada ({len(answer)} caracteres)")
    return {
        "respuesta": answer,
        "citaciones": documentos_relacionados,
        "documentos_encontrados": True,
    }


def resumir_contexto_recuperado(documentos_relacionados: list, pregunta: str = "") -> str:
    """Construye una respuesta detallada a partir del contexto recuperado."""
    if not documentos_relacionados:
        return "No se encontró información relevante en los documentos técnicos."

    # Agrupar por fuente y página para ordenar la información
    secciones = []
    contenidos_vistos = set()
    for doc in documentos_relacionados:
        metadata = getattr(doc, "metadata", {})
        fuente = metadata.get("source", "Documento")
        if fuente:
            fuente = fuente.split("\\")[-1].split("/")[-1]
        pagina = metadata.get("page", 0)
        contenido = getattr(doc, "page_content", str(doc)).strip()
        
        # Evitar contenido duplicado
        if contenido[:100] in contenidos_vistos:
            continue
        contenidos_vistos.add(contenido[:100])
        
        secciones.append(f"📄 {fuente} (p. {pagina}):\n{contenido}")

    if not secciones:
        return "No se encontró información relevante en los documentos técnicos."

    contexto_unido = "\n\n".join(secciones)
    
    return (
        f"Según la información disponible en los documentos técnicos:\n\n"
        f"{contexto_unido}\n\n"
        f"---\n"
        f"*Respuesta generada a partir de los fragmentos recuperados. "
        f"Para una respuesta más detallada y precisa, configura la API key de Groq.*"
    )


# ============================================================================
# WORKFLOW
# ============================================================================

def nodo_triaje(state: AgentState) -> dict:
    """Nodo que realiza triaje de la pregunta."""
    return {"triaje": triaje(state["pregunta"])}


def nodo_auto_resolver(state: AgentState, retriever) -> dict:
    """Nodo que intenta resolver usando RAG."""
    respuesta_rag = busqueda_de_respuestas_RAG(state["pregunta"], retriever)
    rag_exito = respuesta_rag["documentos_encontrados"]
    return {
        "respuesta": respuesta_rag["respuesta"],
        "citaciones": respuesta_rag["citaciones"],
        "rag_exito": rag_exito,
        "accion_final": "AUTO_RESOLVER" if rag_exito else "PEDIR_INFO",
    }


def nodo_pedir_info(state: AgentState) -> dict:
    """Nodo que pide más información al usuario."""
    return {
        "respuesta": "Necesito más información sobre tu consulta. ¿Puedes proporcionar detalles adicionales? Por ejemplo, especifica el cultivo, la plaga o enfermedad, o el tipo de producto que buscas.",
        "citaciones": [],
        "accion_final": "PEDIR_INFO",
    }


def nodo_abrir_ticket(state: AgentState) -> dict:
    """Nodo que simula la apertura de un ticket de soporte."""
    tri = state["triaje"]
    return {
        "respuesta": f"Se abrirá un ticket de soporte con urgencia {tri['urgencia']}. Consulta: {state['pregunta']}",
        "citaciones": [],
        "accion_final": "ABRIR_TICKET",
    }


def arista_decision_triaje(state: AgentState) -> str:
    """Decide el siguiente nodo basado en el triaje inicial."""
    tri = state["triaje"]
    if tri["decision"] == "AUTO_RESOLVER":
        return "rag"
    if tri["decision"] == "PEDIR_INFO":
        return "info"
    return "ticket"


def arista_decision_rag(state: AgentState) -> str:
    """Decide qué hacer si la búsqueda RAG no resolvió la consulta."""
    if state["rag_exito"]:
        return "ok"

    keywords = [
        "recomendar",
        "asesoría",
        "asesoria",
        "urgente",
        "emergencia",
        "ayuda",
        "necesito",
        "problema",
        "ticket",
    ]
    if any(keyword in state["pregunta"].lower() for keyword in keywords):
        return "ticket"
    return "info"


def crear_workflow(retriever):
    """Crea el grafo LangGraph del agente."""
    workflow = StateGraph(AgentState)
    workflow.add_node("triaje", nodo_triaje)
    workflow.add_node("auto_resolver", lambda state: nodo_auto_resolver(state, retriever))
    workflow.add_node("pedir_info", nodo_pedir_info)
    workflow.add_node("abrir_ticket", nodo_abrir_ticket)

    workflow.add_edge(START, "triaje")
    workflow.add_conditional_edges(
        "triaje",
        arista_decision_triaje,
        {"rag": "auto_resolver", "info": "pedir_info", "ticket": "abrir_ticket"},
    )
    workflow.add_conditional_edges(
        "auto_resolver",
        arista_decision_rag,
        {"info": "pedir_info", "ticket": "abrir_ticket", "ok": END},
    )
    workflow.add_edge("pedir_info", END)
    workflow.add_edge("abrir_ticket", END)

    return workflow.compile()


# ============================================================================
# INTERFAZ CLI
# ============================================================================

import argparse


def ejecutar_consulta(pregunta: str, grafo) -> Dict:
    """Ejecuta una consulta a través del agente RAG."""
    return grafo.invoke({"pregunta": pregunta})


def main():
    """Punto de entrada principal del sistema."""
    parser = argparse.ArgumentParser(
        description="AgronomIA - Agente RAG para cultivo de cannabis medicinal"
    )
    parser.add_argument(
        "pregunta",
        nargs="?",
        default="¿Qué producto controla Botrytis?",
        help="Pregunta a evaluar",
    )
    args = parser.parse_args()

    print("\n" + "="*70)
    print("🌾 AgronomIA - Asistente Técnico en Cultivo de Cannabis Medicinal")
    print("="*70)

    log_info("="*70)
    log_info("Iniciando AgronomIA")
    log_info("="*70)

    # Validar pregunta
    error_validacion = validar_pregunta(args.pregunta)
    if error_validacion:
        print(f"\nValidación fallida: {error_validacion}")
        log_error(f"Validación fallida: {error_validacion}")
        return

    try:
        # Cargar documentos y crear retriever
        print("\n📚 Cargando documentos...")
        documentos = cargar_documentos(DOCS_DIR)
        print("\n🔄 Creando retriever...")
        retriever = crear_retriever(documentos)

        # Crear workflow
        print("⚙ Inicializando flujo de trabajo...\n")
        log_info("Workflow inicializado")
        grafo = crear_workflow(retriever)

        # Ejecutar consulta
        print(f"❓ Pregunta: {args.pregunta}\n")
        log_info(f"Ejecutando consulta: {args.pregunta}")
        resultado = ejecutar_consulta(args.pregunta, grafo)

        # Mostrar resultados
        print("\n" + "-"*70)
        print("📊 Triaje:")
        print(f"   Decisión: {resultado['triaje']['decision']}")
        print(f"   Urgencia: {resultado['triaje']['urgencia']}")

        print("\n💬 Respuesta:")
        print(f"   {resultado['respuesta']}")

        if resultado.get("citaciones"):
            citaciones = extraer_citaciones(resultado["citaciones"])
            print("📖 " + formatear_citaciones(citaciones).lstrip("\n").replace("\n", "\n"))

        print("\n" + "="*70)
        log_info("Consulta completada exitosamente")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nAsegúrate de que la carpeta 'docs/' contiene los archivos PDF.")
        log_error(f"FileNotFoundError: {e}")
        raise
    except Exception as e:
        print(f"Error al ejecutar el agente: {e}")
        log_error(f"Error en ejecución: {e}")
        raise


if __name__ == "__main__":
    main()