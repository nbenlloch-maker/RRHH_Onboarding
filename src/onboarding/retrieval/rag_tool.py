import os
from google.adk.tools.retrieval import LlamaIndexRetrieval
from llama_index.core import StorageContext, load_index_from_storage, Document, VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

from onboarding.model_config import get_embedding_model

INDEX_DIR = "faiss_index"

def crear_indice_mock_si_no_existe():
    """Genera un índice FAISS automático la primera vez para facilitar las pruebas locales."""
    if not os.path.exists(INDEX_DIR):
        print("Creando índice FAISS local de prueba...")
        embed_model = get_embedding_model()
        d = len(embed_model.get_text_embedding("dimension probe"))
        
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        doc = Document(text="Política de vacaciones: Los empleados tienen 25 días laborables. Dietas: 45 euros diarios para territorio nacional. Trabajo remoto: 2 días a la semana permitidos tras el primer mes.")
        
        index = VectorStoreIndex.from_documents(
            [doc],
            storage_context=storage_context,
            embed_model=embed_model
        )
        index.storage_context.persist(persist_dir=INDEX_DIR)

def get_hr_rag_tool():
    crear_indice_mock_si_no_existe()
    embed_model = get_embedding_model()
    vector_store = FaissVectorStore.from_persist_dir(INDEX_DIR)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, 
        persist_dir=INDEX_DIR
    )
    index = load_index_from_storage(
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    return LlamaIndexRetrieval(
        retriever=index.as_retriever(),
        name="consultar_manual_rrhh",
        description="Busca información oficial sobre vacaciones, dietas, beneficios y políticas internas de la empresa."
    )
