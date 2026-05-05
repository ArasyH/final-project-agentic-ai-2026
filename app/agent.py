from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate 
from config import (
    CHROMA_DB_PATH, GROQ_API_KEY
)


# Embedding model — sama persis dengan yang dipakai ETL
# PENTING: harus model yang sama, beda model = dimensi vektor berbeda = error
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Knowledge base dari hasil ETL
knowledge_base = Chroma(
    collection_name="stock_knowledge_base",
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_PATH,
)

llm = ChatGroq(
    model="gpt-4o-mini",    # model ringan tapi cukup untuk RAG
    api_key=GROQ_API_KEY,
    temperature=0,       # deterministik — penting untuk riset reprodusibel
    max_tokens=512,
)

SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Kamu adalah asisten analis pasar saham Indonesia yang akurat dan jujur.

Aturan wajib:
1. Jawab HANYA berdasarkan data di bagian Konteks.
2. Jika data tidak tersedia di konteks, jawab: "Data tidak tersedia untuk pertanyaan ini."
3. Jangan pernah mengarang angka, harga, atau persentase.
4. Gunakan Bahasa Indonesia yang jelas.

Konteks data pasar saham:
{context}

Pertanyaan pengguna: {question}

Jawaban berdasarkan data:""",
)

def build_rag_chain() -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=knowledge_base.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        ),
        chain_type_kwargs={"prompt": SYSTEM_PROMPT},
        return_source_documents=True,   # wajib untuk ambil contexts ke RAGAS
    )

# Inisialisasi sekali saat import — tidak tiap request
rag_chain = build_rag_chain()