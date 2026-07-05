# -*- coding: utf-8 -*-
"""Agente RAG local para ejecutar en VS Code usando documentos PDF desde la carpeta docs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
ENV_FILE = ROOT_DIR / ".env"

load_dotenv(ENV_FILE)


class TriajeOut(BaseModel):
    decision: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_TICKET"]
    urgencia: Literal["BAJA", "MEDIANA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    pregunta: str
    triaje: dict
    respuesta: Optional[str]
    citaciones: Optional[list]
    documentos_encontrados: Optional[bool]
    rag_exito: bool
    accion_final: str


class EmbeddingsSimples(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(sum(ord(ch) for ch in text.lower()) % 1000)] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float(sum(ord(ch) for ch in text.lower()) % 1000)]


PROMPT_TRIAJE = """
Eres un especialista en triaje del Service Desk para políticas internas.
Dado el mensaje del usuario, devuelve SOLO un JSON con este formato:
{
  "decision": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_TICKET",
  "urgencia": "BAJA" | "MEDIANA" | "ALTA",
  "campos_faltantes": ["..."]
}
Reglas:
- AUTO_RESOLVER: preguntas claras sobre reglas o procedimientos descritos en las políticas.
- PEDIR_INFO: mensajes imprecisos o sin contexto suficiente.
- ABRIR_TICKET: solicitudes de excepciones, autorizaciones, aprobaciones o acceso especial.
Responde solo con JSON válido.
"""


def cargar_llm() -> Optional[ChatGroq]:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("GROQ_API_KEY no configurada; se usará un modo fallback local.")
        return None

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=groq_api_key,
    )


llm = cargar_llm()


def triaje(mensaje: str) -> Dict:
    if llm is None:
        mensaje_low = mensaje.lower()
        if any(palabra in mensaje_low for palabra in ["excepción", "autoriz", "aprobar", "acceso", "abrir ticket", "ticket"]):
            decision = "ABRIR_TICKET"
            urgencia = "ALTA"
        elif any(palabra in mensaje_low for palabra in ["cómo", "puedo", "política", "reembolso", "vacaciones", "comidas"]):
            decision = "AUTO_RESOLVER"
            urgencia = "BAJA"
        else:
            decision = "PEDIR_INFO"
            urgencia = "MEDIANA"

        return {
            "decision": decision,
            "urgencia": urgencia,
            "campos_faltantes": [],
        }

    respuesta = llm.invoke(
        [
            SystemMessage(content=PROMPT_TRIAJE),
            HumanMessage(content=mensaje),
        ]
    )
    texto = respuesta.content.strip()

    try:
        datos = json.loads(texto)
    except json.JSONDecodeError:
        datos = {
            "decision": "PEDIR_INFO",
            "urgencia": "MEDIA",
            "campos_faltantes": ["No se pudo interpretar la respuesta del modelo"],
        }

    return {
        "decision": datos.get("decision", "PEDIR_INFO"),
        "urgencia": datos.get("urgencia", "MEDIA"),
        "campos_faltantes": datos.get("campos_faltantes", []),
    }


def cargar_documentos(docs_dir: Path) -> List:
    if not docs_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de documentos: {docs_dir}")

    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No se encontraron archivos PDF en: {docs_dir}")

    documentos = []
    for pdf_path in pdf_files:
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            documentos.extend(loader.load())
            print(f"Archivo cargado: {pdf_path.name}")
        except Exception as exc:
            print(f"No se pudo cargar {pdf_path.name}: {exc}")

    if not documentos:
        raise ValueError("No fue posible cargar ningún documento PDF.")

    return documentos


def crear_retriever(documentos: List):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(documentos)

    try:
        embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as exc:
        print(f"No se pudo cargar FastEmbedEmbeddings; se usará un fallback simple: {exc}")
        embeddings = EmbeddingsSimples()

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 1},
    )


def construir_prompt_rag(pregunta: str, documentos_relacionados: List) -> List:
    contexto = "\n\n".join(doc.page_content for doc in documentos_relacionados)
    return [
        SystemMessage(
            content=(
                "Eres un especialista en RR.HH. de la empresa Carraro Desarrollo de Software. "
                "Responde usando únicamente la información del contexto proporcionado. "
                "Si no hay información suficiente, responde exactamente: No lo sé."
            )
        ),
        HumanMessage(content=f"Contexto:\n{contexto}\n\nPregunta del empleado: {pregunta}"),
    ]


# Carga inicial de documentos
DOCUMENTOS = cargar_documentos(DOCS_DIR)
RETRIEVER = crear_retriever(DOCUMENTOS)


def busqueda_de_respuestas_RAG(pregunta: str) -> Dict:
    documentos_relacionados = RETRIEVER.invoke(pregunta)

    if not documentos_relacionados:
        return {
            "respuesta": "No lo sé",
            "citaciones": [],
            "documentos_encontrados": False,
        }

    if llm is None:
        contexto = "\n\n".join(doc.page_content for doc in documentos_relacionados)
        answer = f"He encontrado información relevante en los documentos. Resumen: {contexto[:600]}"
    else:
        respuesta = llm.invoke(construir_prompt_rag(pregunta, documentos_relacionados))
        answer = getattr(respuesta, "content", str(respuesta)).strip()

    if answer.rstrip(".?!") == "No lo sé":
        return {
            "respuesta": "No lo sé",
            "citaciones": [],
            "documentos_encontrados": False,
        }

    return {
        "respuesta": answer,
        "citaciones": documentos_relacionados,
        "documentos_encontrados": True,
    }


def nodo_triaje(state: AgentState) -> AgentState:
    return {"triaje": triaje(state["pregunta"])}


def nodo_auto_resolver(state: AgentState) -> AgentState:
    respuesta_RAG = busqueda_de_respuestas_RAG(state["pregunta"])
    update: AgentState = {
        "respuesta": respuesta_RAG["respuesta"],
        "citaciones": respuesta_RAG["citaciones"],
        "rag_exito": respuesta_RAG["documentos_encontrados"],
    }

    update["accion_final"] = "AUTO_RESOLVER" if respuesta_RAG["documentos_encontrados"] else "pedir_info"
    return update


def nodo_pedir_info(state: AgentState) -> AgentState:
    return {
        "respuesta": "Necesito más información sobre tu pedido.",
        "citaciones": [],
        "accion_final": "PEDIR_INFO",
    }


def nodo_abrir_ticket(state: AgentState) -> AgentState:
    tri = state["triaje"]
    return {
        "respuesta": f"Abrir ticket con urgencia {tri['urgencia']}. Pedido: {state['pregunta']}.",
        "citaciones": [],
        "accion_final": "ABRIR_TICKET",
    }


def arista_decision_triaje(state: AgentState) -> str:
    tri = state["triaje"]
    if tri["decision"] == "AUTO_RESOLVER":
        return "rag"
    if tri["decision"] == "PEDIR_INFO":
        return "info"
    return "ticket"


def arista_decision_rag(state: AgentState) -> str:
    if state["rag_exito"]:
        return "ok"

    keywords = ["aprobación", "aprobar", "excepción", "liberación", "autorización", "autorizar", "abrir ticket", "acceso especial"]
    if any(keyword in state["pregunta"].lower() for keyword in keywords):
        return "ticket"
    return "info"


from langgraph.graph import END, START, StateGraph


workflow = StateGraph(AgentState)
workflow.add_node("triaje", nodo_triaje)
workflow.add_node("auto_resolver", nodo_auto_resolver)
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

grafo = workflow.compile()


def ejecutar_consulta(pregunta: str) -> Dict:
    return grafo.invoke({"pregunta": pregunta})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ejecuta el agente RAG sobre los PDFs de la carpeta docs")
    parser.add_argument("pregunta", nargs="?", default="¿Puedo obtener un reembolso por el internet de mi home office?", help="Pregunta a evaluar")
    args = parser.parse_args()

    print(f"\nPregunta: {args.pregunta}")
    try:
        resultado = ejecutar_consulta(args.pregunta)
    except Exception as exc:
        print(f"Error al ejecutar el agente: {exc}")
        raise

    print(f"Decisión de triaje: {resultado['triaje']['decision']} | urgencia: {resultado['triaje']['urgencia']}")
    print(f"Respuesta: {resultado['respuesta']}")
    if resultado.get("citaciones"):
        print("Citas encontradas:")
        for i, citacion in enumerate(resultado["citaciones"], start=1):
            print(f"- {i}. {citacion.page_content[:180].replace(chr(10), ' ')}")
