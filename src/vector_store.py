"""Vector store module using ChromaDB for semantic retrieval."""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStoreManager:
    """Manages vector storage and retrieval using ChromaDB."""

    def __init__(self, persist_directory: str = "./chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.embeddings = SentenceTransformer(model_name)
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self._collection = None

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts using sentence-transformers."""
        return self.embeddings.encode(texts, convert_to_list=True).tolist()

    def create_collection(self, collection_name: str = "knowledge_base"):
        """Create or get a collection."""
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "KnowledgeNexus document storage"}
        )
        return self._collection

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to the vector store."""
        if not self._collection:
            self.create_collection()

        texts = [chunk['text'] for chunk in chunks]
        ids = [f"doc_{i}_{chunk['metadata'].get('file_name', 'unknown')}"
               for i, chunk in enumerate(chunks)]
        metadatas = [chunk['metadata'] for chunk in chunks]

        embeddings = self._embed_texts(texts)

        self._collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def similarity_search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search with optional filtering."""
        if not self._collection:
            self.create_collection()

        query_embedding = self._embed_texts([query])[0]

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )

        return self._format_results(results)

    def similarity_search_with_mmr(
        self,
        query: str,
        n_results: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Perform MMR (Maximal Marginal Relevance) search for diversity."""
        if not self._collection:
            self.create_collection()

        query_embedding = self._embed_texts([query])[0]

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        return self._format_results(results)

    def _format_results(self, results: dict) -> List[Dict[str, Any]]:
        """Format ChromaDB results into readable structure."""
        formatted = []

        if not results or not results.get('documents'):
            return formatted

        for i in range(len(results['documents'][0])):
            formatted.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else 0
            })

        return formatted

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        if not self._collection:
            self.create_collection()

        return {
            'name': self._collection.name,
            'count': self._collection.count(),
            'persist_directory': self.persist_directory
        }

    def clear_collection(self, collection_name: str = "knowledge_base") -> None:
        """Clear all documents from a collection."""
        try:
            self._client.delete_collection(name=collection_name)
            self._collection = None
        except Exception as e:
            print(f"Error clearing collection: {e}")

    def delete_by_file(self, file_name: str) -> int:
        """Delete all chunks from a specific file."""
        if not self._collection:
            self.create_collection()

        try:
            results = self._collection.get(where={"file_name": file_name})
            if results and results['ids']:
                self._collection.delete(ids=results['ids'])
                return len(results['ids'])
        except Exception as e:
            print(f"Error deleting file: {e}")

        return 0