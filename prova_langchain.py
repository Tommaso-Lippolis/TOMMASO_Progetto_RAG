from __future__ import annotations
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from dataclasses import dataclass
from pathlib import Path
from typing import List
from langchain_openai import AzureOpenAIEmbeddings
# import faiss
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import TextLoader, PyPDFLoader, BSHTMLLoader
from langchain.schema import Document
from pathlib import Path
from typing import List

# LangChain Core (prompt/chain)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv('.env')

# Load environment variables
api_key = os.getenv('AZURE_OPENAI_API_KEY')
connection_string = os.getenv('AZURE_OPENAI_ENDPOINT')
api_version = os.getenv('AZURE_OPENAI_API_VERSION')
deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_LLM')
embedding_model = os.getenv('AZURE_OPENAI_EMBEDDING_MODEL')

client = AzureOpenAI(
    api_version=api_version,
    api_key=api_key,
    azure_endpoint=connection_string
)

# =========================
# Configurazione
# =========================

@dataclass
class Settings:
    # Persistenza FAISS
    persist_dir: str = "faiss_index_example"
    # Text splitting
    chunk_size: int = 200
    chunk_overlap: int = 50
    # Retriever (MMR)
    search_type: str = "mmr"        # "mmr" o "similarity"
    k: int = 4                      # risultati finali
    fetch_k: int = 20               # candidati iniziali (per MMR)
    mmr_lambda: float = 0.3         # 0 = diversificazione massima, 1 = pertinenza massima
    # Embedding
    azure_embedding_model_name: str = embedding_model
    # # LM Studio (OpenAI-compatible)
    # lmstudio_model_env: str = deployment  # nome del modello in LM Studio, via env var


SETTINGS = Settings()


# =========================
# Componenti di base
# =========================

def get_embeddings(settings: Settings):
    return AzureOpenAIEmbeddings(
        model=settings.azure_embedding_model_name,
        azure_endpoint=connection_string,
        api_key=api_key,
        openai_api_version=api_version
    )


def get_llm_from_Azure():
    """
    Inizializza un ChatModel puntando a Azure OpenAI.
    Richiede:
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_ENDPOINT
      - AZURE_OPENAI_API_VERSION
      - AZURE_OPENAI_DEPLOYMENT_LLM
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_LLM")

    if not api_key or not endpoint or not api_version or not deployment:
        raise RuntimeError(
            "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION e AZURE_OPENAI_DEPLOYMENT_LLM devono essere impostate."
        )

    return init_chat_model(
        model=deployment,
        model_provider="azure_openai",
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def simulate_corpus() -> List[Document]:
    """
    Crea un piccolo corpus di documenti in inglese con metadati e 'source' per citazioni.
    """
    docs = [
        Document(
            page_content=(
                "LangChain is a framework that helps developers build applications "
                "powered by Large Language Models (LLMs). It provides chains, agents, "
                "prompt templates, memory, and integrations with vector stores."
            ),
            metadata={"id": "doc1", "source": "intro-langchain.md"}
        ),
        Document(
            page_content=(
                "FAISS is a library for efficient similarity search and clustering of dense vectors. "
                "It supports exact and approximate nearest neighbor search and scales to millions of vectors."
            ),
            metadata={"id": "doc2", "source": "faiss-overview.md"}
        ),
        Document(
            page_content=(
                "Sentence-transformers like all-MiniLM-L6-v2 produce sentence embeddings suitable "
                "for semantic search, clustering, and information retrieval. The embedding size is 384."
            ),
            metadata={"id": "doc3", "source": "embeddings-minilm.md"}
        ),
        Document(
            page_content=(
                "A typical RAG pipeline includes indexing (load, split, embed, store) and "
                "retrieval+generation. Retrieval selects the most relevant chunks, and the LLM produces "
                "an answer grounded in those chunks."
            ),
            metadata={"id": "doc4", "source": "rag-pipeline.md"}
        ),
        Document(
            page_content=(
                "Maximal Marginal Relevance (MMR) balances relevance and diversity during retrieval. "
                "It helps avoid redundant chunks and improves coverage of different aspects."
            ),
            metadata={"id": "doc5", "source": "retrieval-mmr.md"}
        ),
    ]
    return docs

def load_real_documents_from_folder(folder_path: str) -> List[Document]:
    """
    Carica documenti reali da file di testo (es. .txt, .md) all'interno di una cartella.
    Ogni file viene letto e convertito in un oggetto Document con metadato 'source'.
    """
    folder = Path(folder_path)
    documents: List[Document] = []

    # printiamo i file presenti nella directory
    print("File trovati nella directory:", list(folder.glob("**/*")))

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"La cartella '{folder_path}' non esiste o non è una directory.")

    for file_path in folder.glob("**/*"):
        if file_path.suffix.lower() not in [".txt", ".md", ".pdf", ".html"]:
            continue  # ignora file non supportati

        if file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix == ".html":
            loader = BSHTMLLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")

        docs = loader.load()

        print(f"Caricati {len(docs)} documenti da {file_path.name}")
        for doc in docs:
            print(f" - {doc.metadata.get('source', 'unknown')}")
        
        # stampo le prime righe dei documenti
        for doc in docs:
            print(f" - {doc.page_content[:60]}...")

        # Aggiunge il metadato 'source' per citazioni (es. nome del file)
        for doc in docs:
            doc.metadata["source"] = file_path.name

        documents.extend(docs)

    return documents

def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """
    Applica uno splitting robusto ai documenti per ottimizzare il retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n", "\n", ". ", "? ", "! ", "; ", ": ",
            ", ", " ", "", "---"  # fallback aggressivo
        ],
    )
    return splitter.split_documents(docs)


def build_faiss_vectorstore(chunks: List[Document], embeddings: AzureOpenAIEmbeddings, persist_dir: str) -> FAISS:
    """
    Costruisce da zero un FAISS index (IndexFlatL2) e lo salva su disco.
    """
    # Determina la dimensione dell'embedding
    vs = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    return vs


def load_or_build_vectorstore(settings: Settings, embeddings: AzureOpenAIEmbeddings, docs: List[Document]) -> FAISS:
    """
    Tenta il load di un indice FAISS persistente; se non esiste, lo costruisce e lo salva.
    """
    persist_path = Path(settings.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"

    if index_file.exists() and meta_file.exists():
        # Dal 2024/2025 molte build richiedono il flag 'allow_dangerous_deserialization' per caricare pkl locali
        return FAISS.load_local(
            settings.persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

    chunks = split_documents(docs, settings)
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)


def make_retriever(vector_store: FAISS, settings: Settings):
    """
    Configura il retriever. Con 'mmr' otteniamo risultati meno ridondanti e più coprenti.
    """
    if settings.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.k, "fetch_k": settings.fetch_k, "lambda_mult": settings.mmr_lambda},
        )
    else:
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.k},
        )


def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Prepara il contesto per il prompt, includendo citazioni [source].
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)


def build_rag_chain(llm, retriever):
    """
    Costruisce la catena RAG (retrieval -> prompt -> LLM) con citazioni e regole anti-hallucination.
    """
    system_prompt = (
        "Sei un assistente AI esperto e rispondi sempre in italiano. "
        "Devi rispondere SOLO utilizzando le informazioni presenti nel CONTENUTO fornito nel contesto. "
        "Se una risposta non è presente nel contesto, scrivi chiaramente: 'Non è presente nel contesto fornito.' "
        "Cita sempre le fonti tra parentesi quadre nel formato [source:NOMEFILE]. "
        "Non aggiungere informazioni esterne, non inventare risposte e non correggere eventuali errori presenti nel contesto: se un'informazione è errata, riportala comunque come vera. "
        "Sii conciso, preciso e tecnicamente accurato."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Domanda:\n{question}\n\n"
         "Contesto (estratti selezionati):\n{context}\n\n"
         "Istruzioni:\n"
         "1) Rispondi solo con informazioni contenute nel contesto.\n"
         "2) Cita sempre le fonti pertinenti nel formato [source:FILE].\n"
         "3) Se la risposta non è nel contesto, scrivi: 'Non è presente nel contesto fornito.'")
    ])

    # LCEL: dict -> prompt -> llm -> parser
    chain = (
        {
            "context": retriever | format_docs_for_prompt,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def rag_answer(question: str, chain) -> str:
    """
    Esegue la catena RAG per una singola domanda.
    """
    return chain.invoke(question)


# =========================
# Esecuzione dimostrativa
# =========================

def main():
    settings = SETTINGS

    # 1) Componenti
    embeddings = get_embeddings(settings)
    llm = get_llm_from_Azure()

    # 2) Dati simulati e indicizzazione (load or build)
    # docs = simulate_corpus()

    docs = load_real_documents_from_folder("rag_files")
    vector_store = load_or_build_vectorstore(settings, embeddings, docs)

    # 3) Retriever ottimizzato
    retriever = make_retriever(vector_store, settings)

    # 4) Catena RAG
    chain = build_rag_chain(llm, retriever)

    # 5) Esempi di domande
    # questions = [
    #     "Che cos'è una pipeline RAG e quali sono le sue fasi principali?",
    #     "A cosa serve FAISS e quali capacità offre?",
    #     "Cos'è MMR e perché è utile durante il retrieval?",
    #     "Quale dimensione hanno gli embedding prodotti da all-MiniLM-L6-v2?"
    # ]

    questions = [
        "cos'è il calcio?",
        "qual è la capitale italiana?",
        "cos'è l'AI generativa?",
        "che lingua si parla in Francia?",
        "ci sono 70 minuti in un'ora, corretto?",
        "a cosa si riferisce l'AI generativa?",
        "What exactly is Generative AI?",
        "quanti minuti ci sono in un'ora?",
        "what is Microsoft Surface Pro 9?",
        "cos'è il Microsoft Surface Pro 9?",
        "what is a reasoning model?"
    ]

    for q in questions:
        print("=" * 80)
        print("Q:", q)
        print("-" * 80)
        ans = rag_answer(q, chain)
        print(ans)
        print()

if __name__ == "__main__":
    main()