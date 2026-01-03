# 🦜🔗 LangChain Basic - Developing LLM Applications

![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-1.0.3-green.svg)
![UV](https://img.shields.io/badge/uv-latest-orange.svg)
![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-transformers-yellow.svg)

Repositorio del curso **"Developing LLM Applications with LangChain"** de DataCamp. Este proyecto contiene ejemplos prácticos y ejercicios que cubren los fundamentos de desarrollo de aplicaciones con Large Language Models (LLMs) utilizando el framework LangChain.

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Requisitos Previos](#-requisitos-previos)
- [Instalación](#-instalación)
- [Configuración](#️-configuración)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Contenido por Capítulo](#-contenido-por-capítulo)
  - [Chapter 1: Fundamentos de LLMs y Prompts](#chapter-1-fundamentos-de-llms-y-prompts)
  - [Chapter 2: Chains y Agents](#chapter-2-chains-y-agents)
  - [Chapter 3: RAG y Document Processing](#chapter-3-rag-y-document-processing)
- [Uso](#-uso)
- [Dependencias](#-dependencias)
- [Notas Importantes](#-notas-importantes)

---

## 🎯 Descripción

Este repositorio contiene implementaciones prácticas de conceptos clave en el desarrollo de aplicaciones con LLMs:

- **Prompt Engineering**: Plantillas de prompts, few-shot learning, y chat prompts
- **Chains**: Encadenamiento secuencial de operaciones con LLMs
- **Agents**: Agentes ReAct con herramientas personalizadas
- **RAG (Retrieval Augmented Generation)**: Búsqueda semántica y generación aumentada
- **Document Processing**: Carga y procesamiento de documentos (PDF, HTML, CSV)
- **Vector Stores**: Almacenamiento y búsqueda vectorial con Chroma

---

## 🔧 Requisitos Previos

- **Python 3.13** o superior (compatible con 3.11+)
- **uv** - Gestor de paquetes y dependencias de Python (escrito en Rust)
- **OpenAI API Key** - Para usar modelos de OpenAI (GPT-4, GPT-3.5)

### Instalación de `uv`

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# O con pip
pip install uv
```

---

## 📦 Instalación

1. **Clonar el repositorio**

```bash
git clone <tu-repo-url>
cd langchain_basic
```

2. **Crear entorno virtual con Python 3.13**

```bash
# uv creará automáticamente el .venv con la versión especificada en .python-version
uv sync
```

Este comando:

- Lee el archivo `.python-version` (3.13)
- Crea un entorno virtual en `.venv/`
- Instala todas las dependencias del `pyproject.toml`
- Genera/actualiza el `uv.lock` para versiones reproducibles

3. **Instalar dependencias adicionales (requirements.txt)**

```bash
# Algunas dependencias como transformers se instalan por separado
uv pip install -r requirements.txt
```

4. **Verificar la instalación**

```bash
# Verificar que el entorno esté activo
uv run python --version
# Debería mostrar: Python 3.13.x
```

---

## ⚙️ Configuración

### 1. Configurar API Key de OpenAI

Crea un archivo `.env` en la raíz del proyecto:

```bash
cp .env.example .env
```

Edita el archivo `.env` y agrega tu API key:

```env
# .env
OPENAI_API_KEY=<your_api_key_goes_here>
```

> ⚠️ **Importante**: Nunca hagas commit del archivo `.env` (ya está en `.gitignore`)

### 2. Agregar nuevas dependencias (opcional)

Si necesitas agregar más librerías:

```bash
# Agregar una nueva dependencia
uv add nombre-libreria

# Sincronizar (instalar en .venv)
uv sync
```

---

## 📁 Estructura del Proyecto

```
langchain_basic/
│
├── .env                    # Variables de entorno (NO hacer commit)
├── .env.example            # Ejemplo de variables de entorno
├── .python-version         # Versión de Python (3.13)
├── pyproject.toml          # Configuración del proyecto y dependencias
├── uv.lock                 # Lock file con versiones exactas
├── requirements.txt        # Dependencias instaladas con pip (ver nota)
├── README.md               # Este archivo
│
├── Chapter_1/              # Fundamentos de LLMs y Prompts
│   ├── open_ai.py          # Uso básico de OpenAI LLM
│   ├── prompt_template.py  # Plantillas de prompts básicas
│   ├── chat_prompt.py      # Chat prompts con contexto
│   ├── few_shoot.py        # Few-shot learning
│   └── hugging_face.py     # Uso de modelos de Hugging Face (local)
│
├── Chapter_2/              # Chains y Agents
│   ├── sequential_chains.py # Cadenas secuenciales con LCEL
│   ├── tools_intro.py      # Herramientas personalizadas con múltiples ejemplos
│   └── react_intro.py      # Agentes ReAct con Wikipedia
│
└── Chapter_3/              # RAG y Document Processing
    ├── char_splitter.py            # Text splitting por caracteres
    ├── recursive_splitter.py       # Text splitting recursivo
    ├── csv_loader.py               # Cargar documentos CSV (con output detallado)
    ├── html_loader.py              # Cargar documentos HTML
    ├── pdf_loader.py               # Cargar documentos PDF
    ├── docs_splitter.py            # Splitting de documentos HTML
    ├── rag_intro.py                # RAG completo con Chroma
    ├── fifa_countries_audience.csv # Datos de ejemplo (audiencia FIFA)
    ├── rag_vs_fine_tuning.pdf      # PDF de ejemplo para RAG
    └── white_house_executive_order_nov_2023.html # Documento HTML de ejemplo
```

---

## 📚 Contenido por Capítulo

### Chapter 1: Fundamentos de LLMs y Prompts

#### 1️⃣ `open_ai.py` - Uso Básico de OpenAI

Introducción al uso de modelos de OpenAI con LangChain.

```python
# Características:
- Carga de API key desde .env
- Creación de instancia de ChatOpenAI
- Invocación simple de prompts
```

**Conceptos**: LLM básico, invocación de modelos

---

#### 2️⃣ `prompt_template.py` - Plantillas de Prompts

Uso de plantillas dinámicas para generar prompts reutilizables.

```python
# Características:
- PromptTemplate con variables
- Chains con operador LCEL (|)
- Formateo de prompts
```

**Conceptos**: Templates, variables de entrada, LCEL chains

---

#### 3️⃣ `chat_prompt.py` - Chat Prompts

Prompts conversacionales con contexto de sistema, humano y AI.

```python
# Características:
- ChatPromptTemplate con múltiples roles
- Sistema de mensajes (system, human, ai)
- Chains con contexto conversacional
```

**Conceptos**: Chat templates, roles de mensajes, contexto conversacional

---

#### 4️⃣ `few_shoot.py` - Few-Shot Learning

Aprendizaje con ejemplos para guiar las respuestas del LLM.

```python
# Características:
- FewShotPromptTemplate
- Lista de ejemplos estructurados
- Formateo de ejemplos
```

**Conceptos**: Few-shot learning, prompt engineering, ejemplos estructurados

---

#### 5️⃣ `hugging_face.py` - Modelos de Hugging Face

Uso de modelos open-source desde Hugging Face Hub ejecutados **localmente**.

```python
# Características:
- HuggingFacePipeline para modelos locales
- Modelo ligero: crumb/nano-mistral
- Configuración de parámetros de generación (max_new_tokens)
- Sin necesidad de API key (ejecución local)
```

**Conceptos**: Modelos open-source, Hugging Face, pipelines locales, inferencia sin API

**Dependencias requeridas**: `langchain-huggingface`, `transformers`

---

### Chapter 2: Chains y Agents

#### 1️⃣ `sequential_chains.py` - Cadenas Secuenciales

Encadenamiento de múltiples prompts de forma secuencial.

```python
# Características:
- LCEL (LangChain Expression Language)
- Múltiples prompts encadenados
- StrOutputParser para parsear salidas
- Paso de variables entre etapas
```

**Conceptos**: Sequential chains, LCEL, output parsers, composición de prompts

---

#### 2️⃣ `tools_intro.py` - Herramientas Personalizadas

Creación de herramientas personalizadas para agentes con múltiples ejemplos de uso.

```python
# Características:
- Decorador @tool para definir herramientas
- Funciones Python como herramientas para agentes
- Integración con pandas DataFrames
- PromptTemplate para formatear queries
- Múltiples invocaciones de ejemplo (Tech Innovations LLC, Peak Performance Co.)
```

**Conceptos**: Custom tools, function calling, agentes con datos externos, reutilización de prompts

---

#### 3️⃣ `react_intro.py` - Agentes ReAct

Agentes que razonan y actúan con herramientas externas.

```python
# Características:
- create_agent (ReAct pattern)
- Wikipedia tool para búsqueda de información
- HumanMessage para invocación idiomática de LangChain
- Parámetros de reproducibilidad (temperature=0.1, seed=42)
- Razonamiento paso a paso
```

**Conceptos**: ReAct agents, tool usage, reasoning & acting, Wikipedia integration, reproducibilidad

---

### Chapter 3: RAG y Document Processing

#### 1️⃣ `char_splitter.py` - Character Text Splitter

Splitting de texto por número de caracteres.

```python
# Características:
- CharacterTextSplitter
- Control de chunk_size y chunk_overlap
- Separadores personalizados
```

**Conceptos**: Text splitting, chunks, overlap

---

#### 2️⃣ `recursive_splitter.py` - Recursive Character Splitter

Splitting recursivo con jerarquía de separadores.

```python
# Características:
- RecursiveCharacterTextSplitter
- Jerarquía de separadores ["\n\n", "\n", " ", ""]
- Preservación de contexto semántico
```

**Conceptos**: Recursive splitting, semantic chunking, separator hierarchy

---

#### 3️⃣ `csv_loader.py` - CSV Document Loader

Carga de datos estructurados desde archivos CSV con output detallado.

```python
# Características:
- CSVLoader para archivos tabulares
- Rutas absolutas con Path(__file__) para ejecución independiente
- Output formateado mostrando las primeras 3 filas
- Visualización de content y metadata separados
```

**Conceptos**: Document loaders, structured data, file loading, metadata extraction

**Datos**: `fifa_countries_audience.csv` - Datos de audiencia de países FIFA

---

#### 4️⃣ `html_loader.py` - HTML Document Loader

Carga y parsing de documentos HTML.

```python
# Características:
- UnstructuredHTMLLoader
- Extracción de texto de HTML
- Metadata de documentos
```

**Conceptos**: HTML parsing, unstructured data, metadata extraction

**Datos**: `white_house_executive_order_nov_2023.html` - Orden Ejecutiva de IA

---

#### 5️⃣ `pdf_loader.py` - PDF Document Loader

Carga de documentos PDF con metadata.

```python
# Características:
- PyPDFLoader
- Extracción de texto por páginas
- Metadata (página, fuente)
```

**Conceptos**: PDF processing, page extraction, document metadata

---

#### 6️⃣ `docs_splitter.py` - Document Splitting

Splitting de documentos HTML cargados con separadores optimizados.

```python
# Características:
- Carga de HTML con UnstructuredHTMLLoader
- RecursiveCharacterTextSplitter en documentos
- Jerarquía de separadores optimizada: ['\n\n', '\n', '. ', '.', ' ', '']
- Output formateado con longitud de cada chunk
- Visualización del primer chunk como ejemplo
```

**Conceptos**: Document processing pipeline, splitting loaded documents, separator optimization

---

#### 7️⃣ `rag_intro.py` - RAG Completo (★)

**Pipeline completo de Retrieval Augmented Generation.**

```python
# Características:
- Carga de PDF (PyPDFLoader) - rag_vs_fine_tuning.pdf
- Splitting de documentos (RecursiveCharacterTextSplitter)
- Embeddings con OpenAI (text-embedding-3-small)
- Vector store con Chroma (persistente en directorio actual)
- Retriever con similarity search (k=3 documentos)
- RAG chain con LCEL
- RunnablePassthrough para pasar la pregunta directamente
- Generación de respuestas contextualizadas
```

**Pipeline RAG**:

```
1. Load PDF → 2. Split Text → 3. Create Embeddings → 
4. Store in Vector DB → 5. Retrieve Relevant Chunks → 
6. Generate Answer with Context
```

**Conceptos**: RAG, embeddings, vector stores, Chroma, similarity search, retrieval chains, RunnablePassthrough

**Datos**: `rag_vs_fine_tuning.pdf` - Paper comparando RAG vs Fine-tuning

---

## 🚀 Uso

### Ejecutar Scripts

Gracias a las rutas absolutas implementadas, puedes ejecutar los scripts desde **cualquier ubicación**:

```bash
# Desde la raíz del proyecto
uv run Chapter_1/open_ai.py
uv run Chapter_2/sequential_chains.py
uv run Chapter_3/rag_intro.py

# Desde dentro de un capítulo
cd Chapter_3
uv run csv_loader.py

# Desde cualquier otra ubicación
uv run /ruta/completa/Chapter_1/chat_prompt.py
```

### Ejemplos de Uso por Capítulo

**Chapter 1 - Prompts básicos**

```bash
# Probar diferentes proveedores de LLMs
uv run Chapter_1/open_ai.py
uv run Chapter_1/hugging_face.py

# Experimentar con templates
uv run Chapter_1/prompt_template.py
uv run Chapter_1/chat_prompt.py
uv run Chapter_1/few_shoot.py
```

**Chapter 2 - Chains y Agents**

```bash
# Cadenas secuenciales
uv run Chapter_2/sequential_chains.py

# Agentes con herramientas
uv run Chapter_2/tools_intro.py
uv run Chapter_2/react_intro.py
```

**Chapter 3 - RAG y Documents**

```bash
# Document loaders
uv run Chapter_3/csv_loader.py
uv run Chapter_3/html_loader.py
uv run Chapter_3/pdf_loader.py

# Text splitters
uv run Chapter_3/char_splitter.py
uv run Chapter_3/recursive_splitter.py
uv run Chapter_3/docs_splitter.py

# RAG completo
uv run Chapter_3/rag_intro.py
```

---

## 📦 Dependencias

### Dependencias Principales

```toml
dependencies = [
    "langchain>=1.0.3",              # Framework principal
    "langchain-chroma>=1.0.0",       # Vector store con Chroma
    "langchain-community>=0.4.1",    # Loaders y tools comunitarios
    "langchain-huggingface>=1.2.0",  # Integración con Hugging Face
    "langchain-openai>=1.0.1",       # Integración con OpenAI
    "pandas>=2.3.3",                 # Manipulación de datos
    "pypdf>=6.1.3",                  # Procesamiento de PDFs
    "python-dotenv>=1.2.1",          # Manejo de variables de entorno
    "transformers>=4.57.3",          # Modelos de Hugging Face (PyTorch)
    "unstructured>=0.18.15",         # Procesamiento de docs no estructurados
    "wikipedia>=1.4.0",              # API de Wikipedia para agentes
]
```

### Gestión de Dependencias con `uv`

```bash
# Ver dependencias instaladas
uv pip list

# Agregar nueva dependencia
uv add nombre-paquete

# Instalar todas las dependencias del proyecto
uv sync

# Actualizar dependencias
uv lock --upgrade
```

---

## 📝 Notas Importantes

### Sobre `uv`

- **uv** es un gestor de paquetes de Python escrito en Rust, extremadamente rápido
- Reemplaza `pip`, `pip-tools`, `virtualenv` y `poetry` en uno solo
- `uv sync` instala las dependencias **físicamente** en `.venv/lib/python3.13/site-packages/`
- El `uv.lock` garantiza builds reproducibles (como `package-lock.json` en npm)

### Flujo de Trabajo con `uv`

1. **Agregar dependencia**: `uv add <paquete>` → Actualiza `pyproject.toml`
2. **Sincronizar**: `uv sync` → Instala en `.venv/`
3. **Ejecutar**: `uv run <script.py>` → Ejecuta con el entorno correcto

### `uv add` vs `uv pip install`

⚠️ **Importante**: Existen dos formas de instalar paquetes con `uv`:

| Comando | Comportamiento | Persistencia |
|---------|----------------|---------------|
| `uv add <paquete>` | Agrega al `pyproject.toml` y al `uv.lock` | ✅ Persiste con `uv sync` |
| `uv pip install <paquete>` | Instala directamente en `.venv` | ❌ Se borra con `uv sync` |

**En este proyecto**:
- La mayoría de dependencias están en `pyproject.toml` (instaladas con `uv add`)
- `transformers` y sus dependencias están en `requirements.txt` (instaladas con `uv pip install`)

**¿Por qué?** Algunas librerías como `transformers` pueden tener conflictos de resolución de dependencias con `uv add`. En esos casos, se instalan con `uv pip install` y se documentan en `requirements.txt`.

**Para instalar todo el proyecto**:
```bash
# 1. Instalar dependencias del pyproject.toml
uv sync

# 2. Instalar dependencias adicionales del requirements.txt
uv pip install -r requirements.txt
```

### Rutas Absolutas

Todos los scripts que cargan archivos usan rutas absolutas:

```python
from pathlib import Path

# Obtener directorio del script
script_dir = Path(__file__).parent

# Construir ruta al archivo
file_path = script_dir / "archivo.csv"
```

Esto permite ejecutar scripts desde cualquier ubicación sin errores de ruta.

### Variables de Entorno

- El archivo `.env` **NO** debe hacer commit (ya está en `.gitignore`)
- Usa `.env.example` como plantilla
- Siempre usa `python-dotenv` para cargar variables:

```python
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
```

### Vector Store Persistente

El script `rag_intro.py` genera archivos de base de datos de Chroma:

- `chroma.sqlite3`
- Carpeta `b5...cf8/` (O algo similar, relacionado con el ID del vector store)

Estos archivos están en `.gitignore` y se regeneran automáticamente.

---

## 🎓 Recursos Adicionales

- [Documentación de LangChain](https://python.langchain.com/)
- [UV Documentation](https://github.com/astral-sh/uv)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [DataCamp Course](https://www.datacamp.com/courses/developing-llm-applications-with-langchain)

---

## 📄 Licencia

Este proyecto es material educativo del curso de DataCamp "Developing LLM Applications with LangChain".

---

## 🤝 Contribuciones

Este es un repositorio de aprendizaje personal. Si encuentras errores o mejoras, siéntete libre de abrir un issue.

---

**Desarrollado con ❤️ usando LangChain y UV**
