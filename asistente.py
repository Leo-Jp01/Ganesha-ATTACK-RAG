import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

# 1. CARGAR CONFIGURACIÓN
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not token:
    print("ERROR: No se encontró el HUGGINGFACEHUB_API_TOKEN en el archivo .env")
    exit()

# Rutas
RUTA_MITRE = "./attack-pattern"
PERSIST_DIR = "./db_mitre"

# 2. CONFIGURAR EMBEDDINGS
print("---Cargando modelo de Embeddings... ---")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. LÓGICA DE BASE DE DATOS (CREAR O CARGAR)
if not os.path.exists(PERSIST_DIR):
    print("--- La base de datos no existe. Creándola desde los JSON de MITRE... ---")
    
    # Cargador de JSON especializado en técnicas de MITRE, incluyendo detección
    loader = DirectoryLoader(
        path=RUTA_MITRE,
        glob="*.json",
        loader_cls=JSONLoader,
        loader_kwargs={
            "jq_schema": """
            .objects[]
            | select(.type == "attack-pattern")
            | "Técnica: " + .name
            + " | Descripción: " + .description
            + " | Detección: " + (.x_mitre_detection // "N/A")
            """,
            "text_content": True
        }
    )
    
    documentos = loader.load()
    print(f"Documentos cargados: {len(documentos)}")

    # Dividir el texto en fragmentos (Chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = text_splitter.split_documents(documentos)
    print(f"Fragmentos creados: {len(chunks)}")

    # Crear la base de datos vectorial
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print(f"Base de datos guardada en {PERSIST_DIR}")
else:
    print("---Cargando base de datos existente (Modo Rápido)... ---")
    vector_db = Chroma(
        persist_directory=PERSIST_DIR, 
        embedding_function=embeddings
    )

# 4. CONFIGURAR EL CEREBRO (LLM EN MODO CHAT)
print("---Conectando con el LLM (Zephyr-7B via Hugging Face)... ---")

# Motor base del modelo
llm_engine = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=token,
    temperature=0.1,
    max_new_tokens=512,
)

# Envoltorio de Chat para evitar errores de "Conversational Task"
llm = ChatHuggingFace(llm=llm_engine)

# 5. FUNCIÓN DE CONSULTA (RAG)
def generar_ticket_mitre(pregunta_usuario):
    print(f"\nBuscando contexto en la base de datos...")
    
    # Recuperar los 5 fragmentos más relevantes para mayor cobertura
    docs = vector_db.similarity_search(pregunta_usuario, k=5)
    contexto = "\n\n".join([d.page_content for d in docs])
    
    # Construir el prompt para un modelo de Chat
    prompt_final = f"""
Eres un experto analista de ciberseguridad del SOC.
Tu tarea es redactar un ticket de incidente profesional basado en la información de MITRE proporcionada.

CONTEXTO DE MITRE:
{contexto}

PREGUNTA DEL ANALISTA:
{pregunta_usuario}

REGLAS:
1. Si la información no está en el contexto, puedes inferir basada en MITRE, pero indica claramente que es una inferencia.
2. Usa un tono técnico y formal.
3.Si la pregunta del analista es general (como pedir lista de tácticas), responde con un listado claro. 
Si es específica de incidentes, genera el ticket con telemetría y recomendaciones.
4. Incluye claramente:
   - Señales de telemetría esperadas.
   - Técnicas MITRE involucradas.
   - Recomendaciones prácticas para SOC.

RESPUESTA PROFESIONAL:
"""
    # Enviar como mensaje de chat
    mensajes = [HumanMessage(content=prompt_final)]
    
    print("Generando respuesta con la IA...")
    respuesta = llm.invoke(mensajes)
    return respuesta.content

# 6. EJECUCIÓN
if __name__ == "__main__":
    print("\n---SISTEMA GANESHA ATT&CK LISTO ---")
    
    consulta = "¿Cuáles son las principales tácticas de MITRE ATT&CK?"
    
    try:
        resultado = generar_ticket_mitre(consulta)
        print("\n" + "="*50)
        print("TICKET GENERADO POR LA IA:")
        print("="*50)
        print(resultado)
    except Exception as e:
        print(f"\nError durante la ejecución: {e}")