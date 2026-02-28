from src.data_loader import load_all_documents
from src.vectorstore import ChromaVectorStore
from src.search import RAGSearch

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    store = ChromaVectorStore("chroma_store")
    #store.build_from_documents(docs)
    store.load()
    #print(store.query("What is attention mechanism?", top_k=3))
    rag_search = RAGSearch(vector_store=store)
    query = "What are the pillars of the AWS Well-Architected Framework?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Query:", query)
    print("Summary:", summary)

    #results = store.query("Explain Sustainability pillar", top_k=3)
    #print("DEBUG RETRIEVAL:", results)