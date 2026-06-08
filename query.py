import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# Load embedding model (must match the model used when the index was built)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Load FAISS index
index = faiss.read_index("elden.index")

# Load chunks
with open("elden_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def retrieve(query, k=3):
    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        k
    )

    retrieved_chunks = []

    for idx in indices[0]:
        if idx < 0:
            continue
        chunk = chunks[idx]
        retrieved_chunks.append(f"[Page {chunk['page']}]{chunk['text']}")

    return retrieved_chunks, distances[0][0]


def answer_question(question, context):
    prompt = f"""
Answer ONLY using the context below.

If the answer is not present in the context,
say:

"I don't know based on the provided document."

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


while True:
    question = input("\nAsk a question (or type exit): ")

    if question.lower() == "exit":
        break

    retrieved, score = retrieve(question)

    context = "\n\n".join(retrieved)

    answer = answer_question(
        question,
        context
    )

    print("\nAnswer:\n")
    print(answer)