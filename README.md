# Ganesha-ATTACK-RAG
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![MITRE ATT&CK](https://img.shields.io/badge/MITRE-ATT%26CK-red.svg)
![Made with ❤️ by Leo-Jp01](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F-by%20Leo--Jp01-red.svg)

## Descripción.

Ganesha-ATTACK-RAG es un asistente de inteligencia de amenazas basado en **RAG (Retrieval-Augmented Generation)** construido sobre la base de conocimiento de **MITRE ATT&CK**. Permite a analistas SOC, Red Teamers y estudiantes de ciberseguridad consultar en lenguaje natural técnicas de ataque, tácticas asociadas y recomendaciones de detección, obteniendo respuestas estructuradas y accionables.

## Demo.
> *Coming soon.*

## Arquitectura.

```
ENTRADA:
  Pregunta en lenguaje natural
         │
         ▼
  [ QUERY PROCESSOR ]
         │
         ▼
  [ CHROMADB - Vector Store ]
    (MITRE ATT&CK embeddings)
         │
    Recupera los k fragmentos
    más relevantes (k=5)
         │
         ▼
  [ CONTEXT BUILDER ]
    Técnica | Descripción | Detección
         │
         ▼
  [ LLM - Zephyr-7B (HuggingFace) ]
    ChatHuggingFace via Inference API
         │
         ▼
  [ RESPUESTA ESTRUCTURADA ]
    Técnica · Táctica · Telemetría · Recomendaciones SOC
```

## Características.

- Consultas en lenguaje natural sobre técnicas y tácticas de MITRE ATT&CK.
- Recuperación semántica con embeddings (`all-MiniLM-L6-v2`).
- Respuestas estructuradas con telemetría esperada y recomendaciones SOC.
- Base de datos vectorial persistente con ChromaDB (carga rápida en ejecuciones posteriores).
- Integración con Hugging Face Inference API (no requiere GPU local).

## Conocimientos previos.

- Fundamentos de MITRE ATT&CK (tácticas, técnicas, sub-técnicas)
- Conceptos básicos de RAG y embeddings
- Threat Intelligence y telemetría SOC (SIEM, logs de eventos)

## Instalación.

1) Clone el repositorio:
```bash
git clone https://github.com/Leo-Jp01/Ganesha-ATTACK-RAG.git
cd Ganesha-ATTACK-RAG
```

2) Inicie un entorno virtual:
```bash
python -m venv venv_ganesha
source venv_ganesha/bin/activate        # Linux / macOS
venv_ganesha\Scripts\activate           # Windows
```

3) Instale las dependencias:
```bash
pip install -r requirements.txt
```

4) Configure su API Key de Hugging Face:
```bash
cp .env.example .env
nano .env
```
Contenido del `.env`:
```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

5) Agregue los archivos JSON de MITRE ATT&CK en la carpeta `attack-pattern/`:
> Puede descargar los datos desde el [repositorio oficial de MITRE CTI](https://github.com/mitre/cti).

## Uso.

Ejecute el asistente con una pregunta directamente en el script:

```bash
python asistente.py
```

Modifique la variable `consulta` en el bloque `__main__` para cambiar la pregunta:

```python
consulta = "¿Cómo puede un atacante hacer escalada de privilegios en Azure?"
```

**Ejemplo de salida:**

```
---SISTEMA GANESHA ATT&CK LISTO---

**Técnica MITRE:** Exploitation for Privilege Escalation (T1068)
**Táctica:** Privilege Escalation
**Descripción breve:** Los adversarios explotan vulnerabilidades de software para ejecutar
código con privilegios elevados, eludiendo restricciones de acceso del sistema operativo.
**Telemetría esperada:**
- Event ID 4672 (Token elevation)
- Event ID 4696 (Access token manipulation)
- Event ID 4698 (Token duplication)
**Recomendaciones SOC:**
- Aplicar principio de mínimo privilegio en aplicaciones y servicios.
- Mantener parches actualizados para mitigar vulnerabilidades conocidas.
- Monitorear manipulación anómala de tokens de acceso.
---FIN---
```

## Estructura.

```
Ganesha-ATTACK-RAG/
│
├── attack-pattern/          # JSONs de técnicas MITRE ATT&CK
│
├── db_mitre/                # Base de datos vectorial ChromaDB (generada automáticamente)
│
├── venv_ganesha/            # Entorno virtual (no incluido en el repo)
│
├── .env                     # API Key de Hugging Face (no incluido en el repo)
├── .env.example             # Plantilla de variables de entorno
├── .gitignore
├── asistente.py             # Script principal — RAG + LLM
├── requirements.txt
└── README.md
```

## Cómo funciona internamente.

1. **Carga de datos:** Los JSON de MITRE ATT&CK son procesados con `jq_schema` para extraer nombre, descripción y campo de detección de cada `attack-pattern`.
2. **Chunking:** Los documentos son divididos en fragmentos de 800 tokens con 80 de solapamiento usando `RecursiveCharacterTextSplitter`.
3. **Embeddings:** Cada fragmento se convierte en vector con `sentence-transformers/all-MiniLM-L6-v2` y se almacena en ChromaDB.
4. **Consulta (RAG):** Ante una pregunta, se recuperan los 5 fragmentos más similares semánticamente.
5. **Generación:** El contexto recuperado y la pregunta se envían al LLM `zephyr-7b-beta` vía Hugging Face Inference API, que genera una respuesta estructurada en formato SOC.


## Referencias Técnicas.

Este proyecto se desarrolló tomando como base la documentación oficial de las siguientes tecnologías:

- [MITRE ATT&CK - Repositorio CTI](https://github.com/mitre/cti)
- [LangChain - ChromaDB Integration](https://python.langchain.com/docs/integrations/vectorstores/chroma/)
- [LangChain - HuggingFace Endpoints](https://python.langchain.com/docs/integrations/llms/huggingface_endpoint/)
- [Hugging Face - sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Hugging Face - HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)

## Disclaimer.

Este proyecto fue desarrollado de forma autónoma con fines educativos y de investigación en ciberseguridad. En puntos específicos del proceso (optimización, refactor de codigo, documentación o depuración) se utilizó asistencia de IA como apoyo técnico. Toda la lógica, estructura y diseño fueron realizados por mí siguiendo las referencias técnicas listadas.

> ⚠️ Este sistema está diseñado exclusivamente para entornos de práctica y aprendizaje. No reemplaza una plataforma de Threat Intelligence empresarial. La IA puede cometer errores, por lo que se recomienda revisar y verificar la información que proporcione.
