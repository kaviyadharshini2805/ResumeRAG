import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY_GOES_HERE")
llm = genai.GenerativeModel("gemini-2.5-flash")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def load_document(file_path):
    reader = PdfReader(file_path)
    text = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)

    return "\n".join(text)


def split_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=3,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


def create_embeddings(chunks):
    return embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")


def build_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def search(question, index, chunks, k=3):
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, k)

    retrieved = []

    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            retrieved.append({
                "score": float(score),
                "text": chunks[idx]
            })

    return retrieved


def generate_answer(question, retrieved_chunks):

    if not retrieved_chunks:
        return "I couldn't find that information in the document."

    context = "\n\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    prompt = f"""
You are an AI Resume Assistant.

Use ONLY the document context below.

Rules:
- Answer only the user's question.
- Ignore unrelated information.
- Do not hallucinate.
- Use headings and bullet points whenever appropriate.
- If the answer is not present, reply:
"I couldn't find that information in the document."

Resume Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.generate_content(prompt)
    return response.text


def main():

    text = load_document("AboutMe.pdf")

    chunks = split_chunks(text)

    #print(f"Chunks Created: {len(chunks)}")

    embeddings = create_embeddings(chunks)

    index = build_index(embeddings)

    while True:

        question = input("\nAsk a question (type exit to stop): ")

        if question.lower() == "exit":
            break

        retrieved_chunks = search(question, index, chunks)

        print("\nRetrieved Chunks")
        print("-" * 80)

        for i, chunk in enumerate(retrieved_chunks, 1):
            #print(f"\nChunk {i} | Score: {chunk['score']:.4f}")
            print(chunk["text"])
            print("-" * 80)

        answer = generate_answer(question, retrieved_chunks)

        print("\nAnswer")
        print("-" * 80)
        print(answer)


if __name__ == "__main__":
    main()