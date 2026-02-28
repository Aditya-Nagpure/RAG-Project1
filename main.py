from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline


# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    chunks = EmbeddingPipeline().chunk_documents(docs)
    chunk_vectors = EmbeddingPipeline().embed_chunks(chunks)
    print(chunk_vectors) 