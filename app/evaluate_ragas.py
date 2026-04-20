import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_groq import ChatGroq
from langchain_community.embeddings import SentenceTransformerEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Setup judge: Groq (konsisten, tidak perlu OpenAI)
groq_llm = ChatGroq(
    model="llama-3.1-pro",  # pakai model besar untuk judge
    api_key=GROQ_API_KEY,
    temperature=0,
)
embeddings = SentenceTransformerEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# Bungkus agar kompatibel dengan RAGAS
ragas_llm   = LangchainLLMWrapper(groq_llm)
ragas_embed = LangchainEmbeddingsWrapper(embeddings)

# Load data dari export Langfuse (format CSV)
# Kolom wajib: question, answer, contexts (list of strings), ground_truth (opsional)
df = pd.read_csv("langfuse_export.csv")

# RAGAS butuh kolom 'contexts' berupa list, bukan string
df["contexts"] = df["contexts"].apply(eval)

dataset = Dataset.from_pandas(df[["question", "answer", "contexts"]])

# Evaluasi
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=ragas_llm,
    embeddings=ragas_embed,
)

result_df = result.to_pandas()
result_df.to_csv("ragas_results.csv", index=False)
print(result_df[["question", "faithfulness", "answer_relevancy"]].to_string())
print(f"\nRata-rata Faithfulness:     {result_df['faithfulness'].mean():.3f}")
print(f"Rata-rata Answer Relevancy: {result_df['answer_relevancy'].mean():.3f}")