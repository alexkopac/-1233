import os
import pymupdf as fitz
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

chromadb_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "pdf_base"
collection = chromadb_client.get_or_create_collection(name=collection_name)

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
DOCS_FOLDER = "./base"
CHAT_MODEL = "qwen2.5"


def extract_pages_from_pdf(pdf_path):
    """Извлекает текст постранично, возвращая список кортежей (номер_страницы, текст)."""
    pages = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text()
                if page_text and page_text.strip():
                    pages.append((page_num, page_text))
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return pages


def build_chunks_with_pages(pages, words_per_chunk=120, overlap=25):
    """Разбивает текст каждой страницы на чанки с привязкой к номеру страницы."""
    chunk_records = []
    step = max(words_per_chunk - overlap, 1)

    for page_num, text in pages:
        words = text.split()
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + words_per_chunk]).strip()
            if chunk:
                chunk_records.append({
                    "text": chunk,
                    "page": page_num,
                })
    return chunk_records


if not os.path.exists(DOCS_FOLDER):
    os.makedirs(DOCS_FOLDER)
    print(f"{DOCS_FOLDER} was created")

files_index = 0
for filename in sorted(os.listdir(DOCS_FOLDER)):
    if not filename.lower().endswith(".pdf"):
        continue

    file_path = os.path.join(DOCS_FOLDER, filename)
    first_chunk_id = f"file_{filename}_0"

    existing = collection.get(ids=[first_chunk_id])
    if existing and existing["ids"]:
        continue

    pages = extract_pages_from_pdf(file_path)
    if not pages:
        print(f"File '{filename}' is empty or has no text layer")
        continue

    chunk_records = build_chunks_with_pages(pages)
    texts = [item["text"] for item in chunk_records]
    vectors = embedding_model.encode(texts).tolist()
    ids = [f"file_{filename}_{i}" for i in range(len(chunk_records))]
    metadatas = [
        {"source": filename, "page": item["page"], "chunk": i}
        for i, item in enumerate(chunk_records)
    ]

    collection.add(
        documents=texts,
        embeddings=vectors,
        ids=ids,
        metadatas=metadatas,
    )
    files_index += 1

print(f"Indexed new files: {files_index} | Total chunks in DB: {collection.count()}")

if collection.count() == 0:
    print("Database is empty. Add PDF files to ./base and run again.")
else:
    while True:
        user_query = input("\nInput your query (or 'exit' to quit): ").strip()
        if not user_query:
            continue
        if user_query.lower() in ("exit", "quit"):
            break

        query_embedding = embedding_model.encode([user_query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(6, collection.count()),
        )

        retrieved_chunks = results["documents"][0]
        sources = results["metadatas"][0]

        context_blocks = []
        print("\nRetrieved sources:")
        for i, (chunk, meta) in enumerate(zip(retrieved_chunks, sources)):
            page_num = meta.get("page", "?")
            preview = chunk[:100].replace("\n", " ")
            print(f"[{i + 1}] {meta['source']} (стр. {page_num}, chunk #{meta['chunk']}): {preview}...")
            context_blocks.append(
                f"Source: {meta['source']} (Page {page_num})\nContent: {chunk}"
            )

        context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""Answer the user's question based ONLY on the PDF excerpts below.
- Extract and summarize any relevant numbers, ratios, temperatures, or steps found in the text.
- Always cite both the source filename and page number (e.g., [document.pdf, p. 2] or [document.pdf, стр. 2]).
- Reply in the same language as the user's question.
- Only if the excerpts contain no related information at all, say that the documents do not cover this topic.

Context:
{context}

Question:
{user_query}"""

        print("\n" + "=" * 40)
        print("AI Answer:")
        print("=" * 40)

        stream = ollama.chat(
            model=CHAT_MODEL,
            options={"temperature": 0.1},
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful document assistant. Synthesize accurate answers strictly from the provided PDF excerpts and cite filenames with page numbers.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
        print()