# 🌾 AgronomIA

**Agente RAG** para consultas técnicas sobre cultivo de cannabis medicinal.

Usa documentos PDF locales, **LangChain**, **LangGraph**, **FAISS**, **FastEmbed**, **Groq (Llama 3.3 70B)** y **Streamlit**.

---

## 📋 Requisitos

- Python 3.10 o superior
- Una **API key de Groq** (gratuita en [console.groq.com](https://console.groq.com)) — opcional, el sistema funciona en modo *fallback local* sin ella
- Archivos PDF en la carpeta `docs/`

---

## 🚀 Instalación local

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/AgronomIA.git
cd AgronomIA

# 2. Crear y activar entorno virtual
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Crear archivo .env con tu API key de Groq
echo GROQ_API_KEY=tu_api_key_aqui > .env
```

> **Nota:** Si no tienes API key de Groq, omite el paso 4. El sistema usará automáticamente el modo *fallback local* (clasificación por palabras clave + resumen de contexto).

---

## 🖥️ Ejecutar en consola (CLI)

```bash
python src/rag_agente.py "¿Qué producto controla Botrytis?"
```

Sin argumento se ejecuta la pregunta por defecto.

---

## 🌐 Ejecutar en Streamlit (App Web)

```bash
streamlit run app.py
```

Esto abre la interfaz web en `http://localhost:8501`.

---

## ☁️ Despliegue en Streamlit Cloud

### Paso 1 — Preparar el repositorio

Asegúrate de que tu repositorio contenga:

```
AgronomIA/
├── app.py                  # ← Archivo principal de Streamlit
├── requirements.txt        # ← Dependencias
├── src/
│   └── rag_agente.py       # ← Núcleo del agente
├── docs/
│   ├── Guía Técnica de Plagas y Enfermedades en Cannabis.pdf
│   └── Manual de Fichas Técnicas.pdf
└── .gitignore
```

> ⚠️ **No subas** el archivo `.env` — contiene tu API key.
> El `.gitignore` ya lo excluye automáticamente.

### Paso 2 — Subir a GitHub

```bash
git add .
git commit -m "AgronomIA listo para Streamlit Cloud"
git push
```

### Paso 3 — Configurar en Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Inicia sesión con tu cuenta de GitHub
3. Haz clic en **"New app"**
4. Selecciona el repositorio, la rama (`main`) y como archivo principal: **`app.py`**
5. Haz clic en **"Deploy"**

### Paso 4 — Configurar la API key (Secrets)

Después del primer despliegue (fallará por falta de API key):

1. En el dashboard de tu app, ve a **⚙️ Settings → Secrets**
2. Agrega el secreto:

```toml
GROQ_API_KEY = "gsk_tu_api_key_aqui"
```

3. La app se reiniciará automáticamente.

---

## 🧠 ¿Cómo funciona?

```
Pregunta → Validación → Triaje (clasificación)
                            │
                ┌───────────┼───────────┐
           AUTO_RESOLVER  PEDIR_INFO  ABRIR_TICKET
                │              │            │
           ┌────┘              │            │
      Búsqueda RAG             │            │
      (FAISS + LLM)            │            │
           │                   │            │
      ┌────┴────┐              │            │
   Éxito     Fallo             │            │
      │         │              │            │
   Respuesta  Pedir Info    Pedir Info   Abrir Ticket
```

1. **Triaje**: Clasifica la pregunta usando Groq (o fallback por palabras clave)
2. **RAG**: Busca los chunks más relevantes en los PDFs usando FAISS
3. **LLM**: Genera una respuesta basada en el contexto encontrado
4. **Citaciones**: Muestra las fuentes exactas de la información

---

## 🔧 Solución de problemas

| Problema | Causa | Solución |
|----------|-------|----------|
| `No se encontraron archivos PDF` | La carpeta `docs/` no existe o está vacía | Coloca archivos `.pdf` en la carpeta `docs/` |
| `FAISS no está disponible` | Dependencia `faiss-cpu` no instalada | `pip install -r requirements.txt` |
| Respuestas en modo "resumen local" | Falta `GROQ_API_KEY` o error de conexión con Groq | Configura la API key en `.env` o en Streamlit Secrets |
| `ModuleNotFoundError` | Faltan dependencias | Verifica `requirements.txt` y reinstala |
| La app no carga en Streamlit Cloud | Secrets no configurados | Agrega `GROQ_API_KEY` en Settings → Secrets |
| Error de permisos al escribir logs | Entorno efímero (Streamlit Cloud) | El sistema detecta automáticamente Streamlit Cloud y solo usa logging por consola |

### Logs

Los logs locales se guardan en `logs/rag_agente.log`. En Streamlit Cloud puedes ver la salida en el panel **"Manage app → Logs"** del dashboard.

---

## 📦 Estructura del proyecto

```
AgronomIA/
├── app.py              # Interfaz Streamlit
├── requirements.txt    # Dependencias
├── .gitignore          # Archivos ignorados por Git
├── README.md           # Esta guía
├── src/
│   └── rag_agente.py   # Núcleo: configuración, logging, RAG, workflow
└── docs/               # PDFs técnicos
    ├── Guía Técnica de Plagas y Enfermedades en Cannabis.pdf
    └── Manual de Fichas Técnicas.pdf
```

---

## 🛠️ Tecnologías

| Componente | Tecnología |
|-----------|-----------|
| LLM | Groq (Llama 3.3-70B) |
| Framework RAG | LangChain + LangGraph |
| Vector Store | FAISS |
| Embeddings | FastEmbed (multilingual) |
| PDF Loader | PyMuPDF |
| UI | Streamlit |

---

## 📄 Licencia

Este proyecto es de código abierto. Úsalo, modifícalo y compártelo libremente.