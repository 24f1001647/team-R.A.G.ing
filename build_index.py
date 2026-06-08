import fitz
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

PDF_PATH = "Grace Given_ The Mythology of Elden Ring.pdf"

# -----------------------------
# Extract text page by page
# -----------------------------

doc = fitz.open(PDF_PATH)

chunks = []

for page_num in range(len(doc)):

    page = doc[page_num]

    text = page.get_text()

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    current_chunk = ""

    for para in paragraphs:

        words = len(
            (current_chunk + para).split()
        )

        # ~400 word chunks
        if words < 400:
            current_chunk += "\n" + para

        else:

            chunks.append({
                "page": page_num + 1,
                "text": current_chunk.strip()
            })

            current_chunk = para

    if current_chunk:

        chunks.append({
            "page": page_num + 1,
            "text": current_chunk.strip()
        })

print(f"Created {len(chunks)} chunks")

# -----------------------------
# Generate embeddings
# -----------------------------

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

texts = [
    c["text"]
    for c in chunks
]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

# -----------------------------
# Build FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(embeddings)

faiss.write_index(
    index,
    "elden.index"
)

# -----------------------------
# Save metadata
# -----------------------------

with open(
    "elden_chunks.pkl",
    "wb"
) as f:
    pickle.dump(
        chunks,
        f
    )

print("Index saved.")