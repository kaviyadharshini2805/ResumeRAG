import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

GEMINI_API_KEY = "YOUR_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")


def load_document(file_path):
    reader = PdfReader(file_path)
    content = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            content.append(text)

    return "\n".join(content)


def split_chunks(text, chunk_size=500, chunk_overlap=50):

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - chunk_overlap

    for i in range(0, len(text), step):
        chunks.append(text[i:i + chunk_size])

    return chunks


def create_embeddings(chunks):
    embeddings = embedding_model.encode(chunks)
    return np.array(embeddings).astype("float32")


def build_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def search(question, index, chunks, k=3):

    question_embedding = embedding_model.encode([question]).astype("float32")

    distances, indices = index.search(question_embedding, k)

    results = []

    for idx in indices[0]:
        if idx != -1:
            results.append(chunks[idx])

    return results


def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
Answer the question only using the context below.

If the answer is not in the context, say:
"I couldn't find that information in the resume."

Context:
{context}

Question:
{question}
"""

    response = llm.generate_content(prompt)
    return response.text


def main():

    text = load_document("resume.pdf")

    chunks = split_chunks(text)

    embeddings = create_embeddings(chunks)

    index = build_index(embeddings)

    while True:

        question = input("Ask a question (type exit to stop): ")

        if question.lower() == "exit":
            break

        retrieved_chunks = search(question, index, chunks)

        answer = generate_answer(question, retrieved_chunks)

        print("\nAnswer:\n", answer)
        print("-" * 50)


if __name__ == "__main__":
    main()