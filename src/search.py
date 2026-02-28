import os
from dotenv import load_dotenv
from src.vectorstore import ChromaVectorStore
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

class RAGSearch:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"),
                            model_name="llama-3.1-8b-instant",
                            temperature=0.1)

    def search_and_summarize(self, query: str, top_k: int = 3):
        results = self.vector_store.query(query, top_k=top_k)

        context = "\n\n".join([r["content"] for r in results])

        response = self.llm.generate_response(query=query, context=context)

        return response

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vector_store.query(query, top_k=top_k)
        texts = [r["content"] for r in results]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)