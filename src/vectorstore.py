from typing import List, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.embedding import EmbeddingPipeline
import os


class ChromaVectorStore:
    def __init__(self, persist_dir="chroma_store", embedding_model="all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.embedding_fn = HuggingFaceEmbeddings(model_name=embedding_model)
        self.db = None

    def build_from_documents(self, documents: List[Any]):
        emb_pipe = EmbeddingPipeline(model_name="all-MiniLM-L6-v2")
        chunks = emb_pipe.chunk_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self.db = Chroma.from_texts(
            texts=texts,
            embedding=self.embedding_fn,
            metadatas=metadatas,
            persist_directory=self.persist_dir,
        )

        self.db.persist()
        print("[INFO] Chroma DB built and saved")

    def load(self):
        self.db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_fn,
        )
        print("[INFO] Loaded Chroma DB")

    def query(self, query: str, top_k: int = 5):
        results = self.db.similarity_search_with_score(query, k=top_k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]