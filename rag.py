import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_document(file_path):
    reader = PdfReader(file_path)
    content = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            content.append(extracted)
    return "\n".join(content)


def split_chunks(res, chunk_size, chunks_overlap):
    if chunks_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks = []
    steps = chunk_size - chunks_overlap
    for i in range(0, len(res), steps):
        chunk = res[i : chunk_size + i]
        chunks.append(chunk)
    return chunks


def create_embeddings(chunks):
    return model.encode(chunks)


def build_index(vectors):
    embeddings = np.array(vectors).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def search(question, model, index, chunks, k=3):
    question_embedding = model.encode([question])
    question_embedding = question_embedding.astype("float32")
    distances, indices = index.search(question_embedding, k)

    retrieved_chunks = []
    for idx in indices[0]:
        if idx != -1 and idx < len(chunks):
            retrieved_chunks.append(chunks[idx])
    return retrieved_chunks


res = load_document("resume.pdf")
chunk_to_pass = split_chunks(res, 500, 50)
vectors = create_embeddings(chunk_to_pass)
index = build_index(vectors)

result = search("Tell me about candidate's internship experience", model, index, chunk_to_pass,)
for i, chunk in enumerate(result, 1):
    print(f"--- Match {i} ---")
    print(chunk.strip())
    print()