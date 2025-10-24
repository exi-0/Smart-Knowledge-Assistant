import os
import torch
import pandas as pd
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from typing import List
from openai import OpenAI

# ==== Load environment ====
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ==== Setup embedding model and Pinecone ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

pinecone = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "employee-support-kb"

# Delete and recreate index
if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
    pinecone.delete_index(INDEX_NAME)
    print(f"Deleted existing index: {INDEX_NAME}")

pinecone.create_index(
    name=INDEX_NAME,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)
print(f"Created new index: {INDEX_NAME}")
kb_index = pinecone.Index(INDEX_NAME)

# ==== File paths ====
data_paths = {
    "complaint_csv": "data/complaint_knowledge.csv",
    "employee_xlsx": "data/employee_data.xlsx",
    "employee_pdf": "data/All_Employee_Report.pdf"
}

# ==== Helper Functions ====
def load_pdf_text(path):
    text = ""
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def load_all_data():
    texts = []

    # CSV
    if os.path.exists(data_paths["complaint_csv"]):
        df_csv = pd.read_csv(data_paths["complaint_csv"])
        for _, row in df_csv.iterrows():
            texts.append(" | ".join(map(str, row.values)))

    # XLSX
    if os.path.exists(data_paths["employee_xlsx"]):
        df_xlsx = pd.read_excel(data_paths["employee_xlsx"])
        for _, row in df_xlsx.iterrows():
            texts.append(" | ".join(map(str, row.values)))

    # PDF
    if os.path.exists(data_paths["employee_pdf"]):
        pdf_text = load_pdf_text(data_paths["employee_pdf"])
        pdf_chunks = [pdf_text[i:i+500] for i in range(0, len(pdf_text), 500)]
        texts.extend(pdf_chunks)

    return texts

def embed_and_index_texts(texts: List[str], batch_size: int = 100):
    ids = [f"doc_{i}" for i in range(len(texts))]
    embeddings = embedding_model.encode(texts, show_progress_bar=True).tolist()

    print(f"Uploading {len(texts)} vectors in batches...")

    for i in range(0, len(texts), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_vectors = embeddings[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]

        records = [
            {
                "id": batch_ids[j],
                "values": batch_vectors[j],
                "metadata": {"text": batch_texts[j]}
            }
            for j in range(len(batch_ids))
        ]

        kb_index.upsert(records)
        print(f"Uploaded batch {i//batch_size + 1}")

def retrieve_relevant_docs(query: str, top_k: int = 5):
    query_vec = embedding_model.encode([query])[0].tolist()
    results = kb_index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    return [match['metadata']['text'] for match in results['matches']]

def generate_response(user_query: str, context_docs: List[str]) -> str:
    context = "\n\n".join(context_docs)
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {user_query}
Answer:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# ==== Main ====
if __name__ == "__main__":
    print("Loading and embedding data...")
    all_texts = load_all_data()
    embed_and_index_texts(all_texts)

    chat_memory = []
    print("\nAssistant is ready! Ask your questions below. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        memory_context = chat_memory[-3:]
        docs = retrieve_relevant_docs(user_input)
        context = memory_context + docs

        answer = generate_response(user_input, context)
        print(f"Assistant: {answer}\n")

        chat_memory.append(user_input)
        chat_memory.append(answer)
        chat_memory = chat_memory[-6:]  # keep 3 recent Q+A pairs
 