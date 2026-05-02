from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

def search(query, top_k=5, model_name="jhgan/ko-sroberta-multitask", collection_name="ftc_decisions"):
    # Load model
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Embed query
    print(f"Embedding query: '{query}'")
    query_vector = model.encode(query).tolist()

    # Initialize Qdrant client
    print("Initializing Qdrant client...")
    client = QdrantClient(path="./qdrant_db")

    # Search
    print(f"Searching for top {top_k} results...")
    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    ).points

    # Print results
    print("\n--- Search Results ---")
    for i, result in enumerate(search_result):
        print(f"\nResult {i+1} (Score: {result.score:.4f}):")

        doc_title = result.payload.get('doc_metadata', {}).get('의결서제목', '제목 없음')
        violation_types = []
        for violator in result.payload.get('doc_metadata', {}).get('피심인정보', []):
            violation_types.append(violator.get('세부위반유형', '알 수 없음'))

        print(f"의결서 제목: {doc_title}")
        print(f"세부위반유형: {', '.join(set(violation_types))}")
        print(f"본문 일부: {result.payload.get('page_content', '')[:200]}...")

if __name__ == "__main__":
    import sys
    query = "사업자단체의 가격 결정 행위"
    if len(sys.argv) > 1:
        query = sys.argv[1]
    search(query)
