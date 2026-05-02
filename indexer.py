import os
import json
import glob
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

def load_data(raw_data_dir="raw_data"):
    all_chunks = []

    # Get all hybrid.json files
    hybrid_files = glob.glob(os.path.join(raw_data_dir, "*_hybrid.json"))

    for h_file in hybrid_files:
        # Construct matching metadata file name
        base_name = h_file.replace("_hybrid.json", "")
        m_file = f"{base_name}_metadata.json"

        if not os.path.exists(m_file):
            print(f"Warning: Metadata file not found for {h_file}")
            continue

        with open(h_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        with open(m_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Combine chunk data with document metadata
        for chunk in chunks:
            chunk_data = {
                "page_content": chunk.get("page_content", ""),
                "chunk_metadata": chunk.get("metadata", {}),
                "doc_metadata": metadata
            }
            all_chunks.append(chunk_data)

    return all_chunks

def create_embeddings(chunks, model_name="jhgan/ko-sroberta-multitask"):
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Extracting texts...")
    texts = [chunk["page_content"] for chunk in chunks]

    print("Encoding texts to vectors...")
    embeddings = model.encode(texts, show_progress_bar=True)

    return embeddings.tolist(), model.get_sentence_embedding_dimension()

def index_to_qdrant(chunks, embeddings, vector_size, collection_name="ftc_decisions"):
    print("Initializing Qdrant client...")
    client = QdrantClient(path="./qdrant_db")

    print(f"Recreating collection '{collection_name}'...")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        ),
    )

    print("Preparing payloads...")
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        payload = {
            "page_content": chunk["page_content"],
            "chunk_metadata": chunk["chunk_metadata"],
            "doc_metadata": chunk["doc_metadata"]
        }

        points.append(
            models.PointStruct(
                id=i,
                vector=vector,
                payload=payload
            )
        )

    print(f"Uploading {len(points)} points to Qdrant...")
    # Batch upload
    batch_size = 100
    for i in tqdm(range(0, len(points), batch_size)):
        client.upsert(
            collection_name=collection_name,
            points=points[i:i+batch_size]
        )

    print("Indexing completed!")
    return client

def main():
    print("Step 1: Loading data...")
    chunks = load_data()
    print(f"Loaded {len(chunks)} chunks.")

    print("\nStep 2: Creating embeddings...")
    embeddings, vector_size = create_embeddings(chunks)
    print(f"Generated {len(embeddings)} vectors with size {vector_size}.")

    print("\nStep 3: Indexing to Qdrant...")
    index_to_qdrant(chunks, embeddings, vector_size)

if __name__ == "__main__":
    main()
